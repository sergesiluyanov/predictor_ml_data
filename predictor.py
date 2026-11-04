"""Загрузка данных MOEX и прогноз следующей дневной доходности."""

from dataclasses import dataclass
from datetime import datetime, timedelta
import json
import os
from pathlib import Path
import re
import threading

from catboost import CatBoostClassifier
from lightgbm import Booster
import numpy as np

from ensemble_signal_model import lightgbm_frame
from feature_pipeline import build_latest_input
from moex_client import fetch_index_last_n_days, fetch_last_n_days
from signal_model import calibrate_probabilities
from tabular_features import CLASS_NAMES, latest_feature_row


TICKER_RE = re.compile(r"^[A-Z0-9]{1,12}$")


class PredictionError(RuntimeError):
    """Понятная пользователю ошибка подготовки прогноза."""


@dataclass(frozen=True)
class StockPrediction:
    ticker: str
    last_date: str
    forecast_date: str
    last_price: float
    predicted_price: float
    predicted_return: float
    history: list[dict]
    model_path: str
    probabilities: dict[str, float] | None = None
    signal: str | None = None
    signal_threshold: float | None = None
    trading_signal: str | None = None
    trading_up_probability: float | None = None
    trading_confidence: float | None = None


def normalize_ticker(value: str) -> str:
    ticker = value.strip().upper()
    if not TICKER_RE.fullmatch(ticker):
        raise PredictionError("Тикер должен содержать только латинские буквы и цифры.")
    return ticker


def find_model(ticker: str) -> Path:
    """Ищет явную, отдельную для тикера или общую модель."""
    explicit_path = os.getenv("MODEL_PATH")
    if explicit_path:
        path = Path(explicit_path).expanduser()
        if not path.is_file():
            raise PredictionError(f"Модель из MODEL_PATH не найдена: {path}")
        return path

    candidates = (
        Path("models") / "stock_direction.cbm",
        Path("stock_direction.cbm"),
        Path("models") / f"stock_model_{ticker}.tflite",
        Path(f"stock_model_{ticker}.tflite"),
        Path("stock_model.tflite"),
    )
    for path in candidates:
        if path.is_file():
            return path

    raise PredictionError(
        "Модель не найдена. Обучите models/stock_direction.cbm, "
        "stock_model.tflite или "
        f"models/stock_model_{ticker}.tflite."
    )


def _next_weekday(date_string: str) -> str:
    result = datetime.strptime(date_string, "%Y-%m-%d") + timedelta(days=1)
    while result.weekday() >= 5:
        result += timedelta(days=1)
    return result.strftime("%Y-%m-%d")


class _ModelRunner:
    """Переиспользует интерпретатор и защищает его от одновременных вызовов."""

    def __init__(self, model_path: Path):
        try:
            import tensorflow as tf
        except ImportError as exc:
            raise PredictionError(
                "Для TFLite-модели требуется TensorFlow. "
                "Используйте общую CatBoost-модель или установите TensorFlow."
            ) from exc
        self.interpreter = tf.lite.Interpreter(model_path=str(model_path))
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()[0]
        self.output_details = self.interpreter.get_output_details()[0]
        self.lock = threading.Lock()

    def predict(self, features: np.ndarray) -> float:
        expected_shape = tuple(int(value) for value in self.input_details["shape"])
        if expected_shape != features.shape:
            raise PredictionError(
                f"Форма входа модели {expected_shape}, ожидалась {features.shape}."
            )

        input_value = features.astype(self.input_details["dtype"])
        with self.lock:
            self.interpreter.set_tensor(self.input_details["index"], input_value)
            self.interpreter.invoke()
            output = self.interpreter.get_tensor(self.output_details["index"])
        return float(np.asarray(output).reshape(-1)[0])


_RUNNERS: dict[Path, _ModelRunner] = {}
_RUNNERS_LOCK = threading.Lock()


def _get_runner(model_path: Path) -> _ModelRunner:
    resolved_path = model_path.resolve()
    with _RUNNERS_LOCK:
        if resolved_path not in _RUNNERS:
            _RUNNERS[resolved_path] = _ModelRunner(resolved_path)
        return _RUNNERS[resolved_path]


class _DirectionModelRunner:
    def __init__(self, model_path: Path):
        self.model = CatBoostClassifier()
        self.model.load_model(model_path)
        metadata_path = model_path.with_suffix(".json")
        if not metadata_path.is_file():
            raise PredictionError(
                f"Не найдены метаданные модели: {metadata_path}"
            )
        self.metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        self.lock = threading.Lock()

    def predict(
        self, ticker: str, rows: list[dict], market_rows: list[dict]
    ) -> tuple[float, dict[str, float], str, float]:
        features = latest_feature_row(rows, ticker, market_rows)
        with self.lock:
            raw_probabilities = self.model.predict_proba(features)[0]
        classes = [int(value) for value in self.model.classes_]
        by_class = {
            class_id: float(raw_probabilities[index])
            for index, class_id in enumerate(classes)
        }
        probabilities = {
            CLASS_NAMES[class_id]: by_class.get(class_id, 0.0)
            for class_id in CLASS_NAMES
        }
        class_returns = self.metadata.get("class_mean_returns", {})
        predicted_return = sum(
            by_class.get(class_id, 0.0)
            * float(class_returns.get(str(class_id), 0.0))
            for class_id in CLASS_NAMES
        )
        signal_id = max(by_class, key=by_class.get)
        signal_threshold = float(
            self.metadata.get("recommended_probability_threshold", 0.55)
        )
        return (
            predicted_return,
            probabilities,
            CLASS_NAMES[signal_id],
            signal_threshold,
        )


_DIRECTION_RUNNERS: dict[Path, _DirectionModelRunner] = {}


def _get_direction_runner(model_path: Path) -> _DirectionModelRunner:
    resolved_path = model_path.resolve()
    with _RUNNERS_LOCK:
        if resolved_path not in _DIRECTION_RUNNERS:
            _DIRECTION_RUNNERS[resolved_path] = _DirectionModelRunner(
                resolved_path
            )
        return _DIRECTION_RUNNERS[resolved_path]


class _BinarySignalRunner:
    def __init__(self, model_path: Path):
        self.model = CatBoostClassifier()
        self.model.load_model(model_path)
        metadata_path = model_path.with_suffix(".json")
        self.metadata = json.loads(
            metadata_path.read_text(encoding="utf-8")
        )
        self.lock = threading.Lock()

    def predict(
        self, ticker: str, rows: list[dict], market_rows: list[dict]
    ) -> tuple[str, float, float]:
        features = latest_feature_row(rows, ticker, market_rows)
        with self.lock:
            raw_probability = float(
                self.model.predict_proba(features)[0, 1]
            )
        up_probability = float(
            calibrate_probabilities(
                np.asarray([raw_probability]),
                float(self.metadata["calibration_coefficient"]),
                float(self.metadata["calibration_intercept"]),
            )[0]
        )
        up_threshold = self.metadata.get("up_confidence_threshold")
        down_threshold = self.metadata.get("down_confidence_threshold")
        if up_threshold is not None and up_probability >= float(up_threshold):
            return "up", up_probability, up_probability
        down_probability = 1.0 - up_probability
        if (
            down_threshold is not None
            and down_probability >= float(down_threshold)
        ):
            return "down", up_probability, down_probability
        return "none", up_probability, max(up_probability, down_probability)


_BINARY_RUNNERS: dict[Path, _BinarySignalRunner] = {}


def _get_binary_runner(model_path: Path) -> _BinarySignalRunner:
    resolved_path = model_path.resolve()
    with _RUNNERS_LOCK:
        if resolved_path not in _BINARY_RUNNERS:
            _BINARY_RUNNERS[resolved_path] = _BinarySignalRunner(
                resolved_path
            )
        return _BINARY_RUNNERS[resolved_path]


class _LightGBMSignalRunner:
    def __init__(self, model_path: Path):
        self.model = Booster(model_file=str(model_path))
        metadata_path = model_path.with_suffix(".json")
        self.metadata = json.loads(
            metadata_path.read_text(encoding="utf-8")
        )
        self.lock = threading.Lock()

    def predict(
        self, ticker: str, rows: list[dict], market_rows: list[dict]
    ) -> tuple[str, float, float]:
        features = latest_feature_row(rows, ticker, market_rows)
        encoded = lightgbm_frame(
            features, self.metadata["ticker_mapping"]
        )
        with self.lock:
            raw_probability = float(self.model.predict(encoded)[0])
        up_probability = float(
            calibrate_probabilities(
                np.asarray([raw_probability]),
                float(self.metadata["calibration_coefficient"]),
                float(self.metadata["calibration_intercept"]),
            )[0]
        )
        policy = self.metadata["stable_warning_policy"]
        market_volatility = float(
            features["market_volatility_20"].iloc[0]
        )
        if market_volatility >= float(
            policy["max_market_volatility_20"]
        ):
            return "none", up_probability, 1.0 - up_probability
        down_probability = 1.0 - up_probability
        if down_probability >= float(
            policy["down_confidence_threshold"]
        ):
            return "down", up_probability, down_probability
        return "none", up_probability, max(up_probability, down_probability)


_LIGHTGBM_RUNNERS: dict[Path, _LightGBMSignalRunner] = {}


def _get_lightgbm_runner(model_path: Path) -> _LightGBMSignalRunner:
    resolved_path = model_path.resolve()
    with _RUNNERS_LOCK:
        if resolved_path not in _LIGHTGBM_RUNNERS:
            _LIGHTGBM_RUNNERS[resolved_path] = _LightGBMSignalRunner(
                resolved_path
            )
        return _LIGHTGBM_RUNNERS[resolved_path]


def predict_stock(ticker_value: str, history_days: int = 730) -> StockPrediction:
    ticker = normalize_ticker(ticker_value)
    model_path = find_model(ticker)

    try:
        rows = fetch_last_n_days(ticker, days=history_days)
    except Exception as exc:
        raise PredictionError(f"Не удалось загрузить данные MOEX: {exc}") from exc

    if not rows:
        raise PredictionError(f"MOEX не вернула данные для тикера {ticker}.")

    market_rows = None
    try:
        if model_path.suffix.lower() == ".cbm":
            market_rows = fetch_index_last_n_days("IMOEX", history_days)
            (
                predicted_return,
                probabilities,
                signal,
                signal_threshold,
            ) = _get_direction_runner(model_path).predict(
                ticker, rows, market_rows
            )
        else:
            try:
                features = build_latest_input(rows)
            except ValueError as exc:
                raise PredictionError(str(exc)) from exc
            predicted_return = _get_runner(model_path).predict(features)
            probabilities = None
            signal = None
            signal_threshold = None

        lightgbm_signal_path = Path("models/stock_signal_lgbm.txt")
        catboost_signal_path = Path("models/stock_signal.cbm")
        if lightgbm_signal_path.is_file() or catboost_signal_path.is_file():
            if market_rows is None:
                market_rows = fetch_index_last_n_days(
                    "IMOEX", history_days
                )
            if lightgbm_signal_path.is_file():
                (
                    trading_signal,
                    trading_up_probability,
                    trading_confidence,
                ) = _get_lightgbm_runner(
                    lightgbm_signal_path
                ).predict(ticker, rows, market_rows)
            else:
                (
                    trading_signal,
                    trading_up_probability,
                    trading_confidence,
                ) = _get_binary_runner(
                    catboost_signal_path
                ).predict(ticker, rows, market_rows)
        else:
            trading_signal = None
            trading_up_probability = None
            trading_confidence = None
    except PredictionError:
        raise
    except Exception as exc:
        raise PredictionError(f"Не удалось выполнить модель: {exc}") from exc

    last_row = rows[-1]
    last_price = float(last_row["close"])
    predicted_price = last_price * (1.0 + predicted_return)
    return StockPrediction(
        ticker=ticker,
        last_date=last_row["tradedate"],
        forecast_date=_next_weekday(last_row["tradedate"]),
        last_price=last_price,
        predicted_price=predicted_price,
        predicted_return=predicted_return,
        history=rows,
        model_path=str(model_path),
        probabilities=probabilities,
        signal=signal,
        signal_threshold=signal_threshold,
        trading_signal=trading_signal,
        trading_up_probability=trading_up_probability,
        trading_confidence=trading_confidence,
    )
