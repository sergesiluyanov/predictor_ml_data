"""Walk-forward-бэктест модели на ещё не виденных временных периодах."""

import argparse
import csv
from datetime import datetime
import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from feature_pipeline import build_supervised_dataset
from moex_client import (
    DEFAULT_LIQUID_TICKERS,
    fetch_last_n_days,
    fetch_liquid_tickers_full,
)
from train_stock_model import create_model


def calculate_metrics(
    actual: np.ndarray,
    predicted: np.ndarray,
    commission_bps: float,
) -> dict:
    error = predicted - actual
    direction_accuracy = float(
        np.mean((predicted >= 0) == (actual >= 0))
    )
    if len(actual) > 1 and np.std(actual) > 0 and np.std(predicted) > 0:
        correlation = float(np.corrcoef(actual, predicted)[0, 1])
    else:
        correlation = 0.0

    # Консервативная long/cash-стратегия: держим акцию только при прогнозе > 0.
    positions = (predicted > 0).astype(np.float32)
    turnover = np.abs(np.diff(positions, prepend=0.0))
    commission = commission_bps / 10_000.0
    net_returns = positions * actual - turnover * commission

    return {
        "count": int(len(actual)),
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(np.square(error)))),
        "direction_accuracy": direction_accuracy,
        "correlation": correlation,
        "long_cash_return": float(np.prod(1.0 + net_returns) - 1.0),
        "active_days": int(np.sum(positions)),
    }


def _fold_starts(
    sample_count: int,
    min_train_samples: int,
    test_samples: int,
    folds: int,
) -> list[int]:
    if sample_count < min_train_samples + test_samples:
        return []
    first = max(min_train_samples, sample_count - folds * test_samples)
    return list(range(first, sample_count, test_samples))[:folds]


def backtest_ticker(
    ticker: str,
    days: int,
    epochs: int,
    min_train_samples: int,
    test_samples: int,
    folds: int,
    commission_bps: float,
) -> tuple[list[dict], dict] | None:
    rows = fetch_last_n_days(ticker, days=days)
    dataset = build_supervised_dataset(rows)
    if dataset is None:
        return None

    starts = _fold_starts(
        len(dataset.y), min_train_samples, test_samples, folds
    )
    if not starts:
        return None

    predictions: list[dict] = []
    for fold_number, test_start in enumerate(starts, start=1):
        test_end = min(test_start + test_samples, len(dataset.y))
        tf.keras.backend.clear_session()
        tf.keras.utils.set_random_seed(42 + fold_number)
        model = create_model()
        model.fit(
            dataset.X[:test_start],
            dataset.y[:test_start],
            epochs=epochs,
            batch_size=32,
            shuffle=False,
            verbose=0,
        )
        model_prediction = model.predict(
            dataset.X[test_start:test_end], verbose=0
        ).reshape(-1)
        momentum_prediction = dataset.X[test_start:test_end, -1, 1]
        mean_return_prediction = np.mean(
            dataset.X[test_start:test_end, -5:, 1], axis=1
        )

        for offset, sample_index in enumerate(
            range(test_start, test_end)
        ):
            predictions.append(
                {
                    "ticker": ticker,
                    "fold": fold_number,
                    "target_date": dataset.target_dates[sample_index],
                    "current_price": float(
                        dataset.current_prices[sample_index]
                    ),
                    "target_price": float(
                        dataset.target_prices[sample_index]
                    ),
                    "actual_return": float(dataset.y[sample_index]),
                    "model_prediction": float(model_prediction[offset]),
                    "zero_prediction": 0.0,
                    "momentum_prediction": float(
                        momentum_prediction[offset]
                    ),
                    "mean5_prediction": float(
                        mean_return_prediction[offset]
                    ),
                }
            )

    actual = np.asarray(
        [row["actual_return"] for row in predictions], dtype=np.float32
    )
    metrics = {}
    for name in ("model", "zero", "momentum", "mean5"):
        predicted = np.asarray(
            [row[f"{name}_prediction"] for row in predictions],
            dtype=np.float32,
        )
        metrics[name] = calculate_metrics(
            actual, predicted, commission_bps
        )
    metrics["buy_hold_return"] = float(np.prod(1.0 + actual) - 1.0)
    metrics["first_test_date"] = predictions[0]["target_date"]
    metrics["last_test_date"] = predictions[-1]["target_date"]
    return predictions, metrics


def _aggregate_metrics(
    predictions: list[dict], commission_bps: float
) -> dict:
    actual = np.asarray(
        [row["actual_return"] for row in predictions], dtype=np.float32
    )
    result = {}
    for name in ("model", "zero", "momentum", "mean5"):
        predicted = np.asarray(
            [row[f"{name}_prediction"] for row in predictions],
            dtype=np.float32,
        )
        result[name] = calculate_metrics(actual, predicted, commission_bps)
        per_ticker_returns = []
        for ticker in sorted({row["ticker"] for row in predictions}):
            ticker_rows = [
                row for row in predictions if row["ticker"] == ticker
            ]
            ticker_actual = np.asarray(
                [row["actual_return"] for row in ticker_rows],
                dtype=np.float32,
            )
            ticker_predicted = np.asarray(
                [row[f"{name}_prediction"] for row in ticker_rows],
                dtype=np.float32,
            )
            per_ticker_returns.append(
                calculate_metrics(
                    ticker_actual, ticker_predicted, commission_bps
                )["long_cash_return"]
            )
        result[name]["mean_ticker_long_cash_return"] = float(
            np.mean(per_ticker_returns)
        )
        result[name]["pooled_long_cash_return"] = result[name][
            "long_cash_return"
        ]
        result[name]["long_cash_return"] = result[name][
            "mean_ticker_long_cash_return"
        ]
    buy_hold_returns = []
    for ticker in sorted({row["ticker"] for row in predictions}):
        ticker_actual = np.asarray(
            [
                row["actual_return"]
                for row in predictions
                if row["ticker"] == ticker
            ],
            dtype=np.float32,
        )
        buy_hold_returns.append(float(np.prod(1.0 + ticker_actual) - 1.0))
    result["mean_ticker_buy_hold_return"] = float(
        np.mean(buy_hold_returns)
    )
    return result


def _save_chart(
    predictions: list[dict],
    output_path: Path,
    commission_bps: float,
) -> None:
    fig, ax = plt.subplots(figsize=(11, 6))
    for name, label in (
        ("model", "Модель"),
        ("momentum", "Последняя доходность"),
        ("mean5", "Средняя доходность за 5 дней"),
    ):
        curves = []
        for ticker in sorted({row["ticker"] for row in predictions}):
            ticker_rows = [
                row for row in predictions if row["ticker"] == ticker
            ]
            actual = np.asarray(
                [row["actual_return"] for row in ticker_rows],
                dtype=np.float32,
            )
            predicted = np.asarray(
                [row[f"{name}_prediction"] for row in ticker_rows],
                dtype=np.float32,
            )
            positions = (predicted > 0).astype(np.float32)
            turnover = np.abs(np.diff(positions, prepend=0.0))
            net_returns = (
                positions * actual
                - turnover * (commission_bps / 10_000.0)
            )
            curves.append(np.cumprod(1.0 + net_returns) - 1.0)
        common_length = min(len(curve) for curve in curves)
        mean_curve = np.mean(
            np.stack([curve[:common_length] for curve in curves]), axis=0
        )
        ax.plot(mean_curve * 100, label=label)
    ax.set_title("Walk-forward: средняя long/cash-доходность по тикерам")
    ax.set_xlabel("Последовательные тестовые прогнозы")
    ax.set_ylabel("Накопленная доходность, %")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=140)
    plt.close(fig)


def _parse_tickers(value: str, max_tickers: int) -> list[str]:
    if value.lower() != "all":
        return [
            ticker.strip().upper()
            for ticker in value.split(",")
            if ticker.strip()
        ]
    tickers = fetch_liquid_tickers_full()
    if not tickers:
        tickers = list(DEFAULT_LIQUID_TICKERS)
    return tickers[:max_tickers]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Walk-forward backtest без утечки будущих данных"
    )
    parser.add_argument("--tickers", default="SBER")
    parser.add_argument("--max-tickers", type=int, default=20)
    parser.add_argument("--days", type=int, default=1500)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--min-train-samples", type=int, default=252)
    parser.add_argument("--test-samples", type=int, default=60)
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--commission-bps", type=float, default=10.0)
    parser.add_argument("--out-dir", default="reports")
    args = parser.parse_args()

    tickers = _parse_tickers(args.tickers, args.max_tickers)
    all_predictions: list[dict] = []
    ticker_metrics = {}
    skipped = []

    for index, ticker in enumerate(tickers, start=1):
        print(f"[{index}/{len(tickers)}] Бэктест {ticker}...")
        try:
            result = backtest_ticker(
                ticker=ticker,
                days=args.days,
                epochs=args.epochs,
                min_train_samples=args.min_train_samples,
                test_samples=args.test_samples,
                folds=args.folds,
                commission_bps=args.commission_bps,
            )
        except Exception as exc:
            skipped.append({"ticker": ticker, "reason": str(exc)})
            continue
        if result is None:
            skipped.append(
                {"ticker": ticker, "reason": "недостаточно истории"}
            )
            continue
        predictions, metrics = result
        all_predictions.extend(predictions)
        ticker_metrics[ticker] = metrics

    if not all_predictions:
        raise SystemExit(f"Нет результатов. Пропущено: {skipped}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"backtest_{timestamp}.csv"
    json_path = output_dir / f"backtest_{timestamp}.json"
    chart_path = output_dir / f"backtest_{timestamp}.png"

    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file, fieldnames=list(all_predictions[0].keys())
        )
        writer.writeheader()
        writer.writerows(all_predictions)

    aggregate = _aggregate_metrics(
        all_predictions, args.commission_bps
    )
    report = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "config": vars(args),
        "tickers_tested": list(ticker_metrics),
        "skipped": skipped,
        "aggregate": aggregate,
        "by_ticker": ticker_metrics,
    }
    with json_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=False, indent=2)
    _save_chart(all_predictions, chart_path, args.commission_bps)

    model = aggregate["model"]
    print(
        f"Прогнозов: {model['count']}; "
        f"direction accuracy: {model['direction_accuracy']:.2%}; "
        f"MAE: {model['mae']:.5f}; "
        f"long/cash: {model['long_cash_return']:.2%}"
    )
    print(f"Отчёт: {json_path}")
    print(f"Данные: {csv_path}")
    print(f"График: {chart_path}")


if __name__ == "__main__":
    # Уменьшает шум TensorFlow в CLI, не скрывая ошибки.
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "1")
    main()
