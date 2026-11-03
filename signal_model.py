"""Обучение и калибровка бинарной модели торгового сигнала."""

from catboost import CatBoostClassifier
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from tabular_features import CATEGORICAL_FEATURES, FEATURE_COLUMNS


def prepare_binary_dataset(
    frame: pd.DataFrame,
    min_abs_return: float,
) -> pd.DataFrame:
    """Оставляет движения, превышающие шум и торговые издержки."""
    result = frame[
        frame["strategy_return"].abs() >= min_abs_return
    ].copy()
    result["signal_target"] = (
        result["strategy_return"] > 0
    ).astype(int)
    return result.reset_index(drop=True)


def _logit(probabilities: np.ndarray) -> np.ndarray:
    clipped = np.clip(probabilities, 1e-6, 1.0 - 1e-6)
    return np.log(clipped / (1.0 - clipped)).reshape(-1, 1)


def calibrate_probabilities(
    raw_probabilities: np.ndarray,
    coefficient: float,
    intercept: float,
) -> np.ndarray:
    logits = _logit(raw_probabilities).reshape(-1)
    calibrated_logits = coefficient * logits + intercept
    return 1.0 / (1.0 + np.exp(-calibrated_logits))


def evaluate_confidence_thresholds(
    probabilities: np.ndarray,
    actual: np.ndarray,
) -> tuple[float, dict]:
    scores = {}
    minimum_signals = max(100, int(len(actual) * 0.02))
    for threshold in (0.55, 0.60, 0.65, 0.70, 0.75):
        signal_mask = (probabilities >= threshold) | (
            probabilities <= 1.0 - threshold
        )
        signal_count = int(signal_mask.sum())
        predicted = (probabilities >= 0.5).astype(int)
        precision = (
            float(np.mean(predicted[signal_mask] == actual[signal_mask]))
            if signal_count
            else 0.0
        )
        scores[str(threshold)] = {
            "precision": precision,
            "coverage": float(signal_mask.mean()),
            "signals": signal_count,
        }

    enough_samples = {
        threshold: values
        for threshold, values in scores.items()
        if values["signals"] >= minimum_signals
    }
    target_precision = {
        threshold: values
        for threshold, values in enough_samples.items()
        if values["precision"] >= 0.55
    }
    if target_precision:
        selected = max(
            target_precision,
            key=lambda threshold: target_precision[threshold]["coverage"],
        )
    elif enough_samples:
        selected = max(
            enough_samples,
            key=lambda threshold: (
                enough_samples[threshold]["precision"],
                enough_samples[threshold]["coverage"],
            ),
        )
    else:
        selected = max(
            scores,
            key=lambda threshold: scores[threshold]["precision"],
        )
    return float(selected), scores


def select_direction_thresholds(
    probabilities: np.ndarray,
    actual: np.ndarray,
    target_precision: float = 0.55,
) -> tuple[float | None, float | None, dict]:
    """Независимо допускает up/down только при достаточной precision."""
    all_scores = {}
    selected = {}
    minimum_signals = max(50, int(len(actual) * 0.01))
    for direction, expected in (("up", 1), ("down", 0)):
        scores = {}
        for threshold in (0.55, 0.60, 0.65, 0.70, 0.75, 0.80):
            mask = (
                probabilities >= threshold
                if direction == "up"
                else probabilities <= 1.0 - threshold
            )
            count = int(mask.sum())
            precision = (
                float(np.mean(actual[mask] == expected)) if count else 0.0
            )
            scores[str(threshold)] = {
                "precision": precision,
                "coverage": float(mask.mean()),
                "signals": count,
            }
        eligible = {
            threshold: values
            for threshold, values in scores.items()
            if values["signals"] >= minimum_signals
            and values["precision"] >= target_precision
        }
        selected[direction] = (
            float(
                max(
                    eligible,
                    key=lambda threshold: eligible[threshold]["coverage"],
                )
            )
            if eligible
            else None
        )
        all_scores[direction] = scores
    return selected["up"], selected["down"], all_scores


def train_binary_model(
    frame: pd.DataFrame,
    iterations: int,
    depth: int,
    learning_rate: float,
) -> tuple[CatBoostClassifier, dict]:
    """Временные split: train / early-stop / calibration / threshold."""
    dates = np.asarray(sorted(frame["target_date"].unique()))
    if len(dates) < 120:
        raise ValueError("Недостаточно дат для бинарной модели.")
    boundaries = [
        dates[int(len(dates) * fraction)]
        for fraction in (0.65, 0.75, 0.85)
    ]
    train = frame[frame["target_date"] < boundaries[0]]
    early_stop = frame[
        (frame["target_date"] >= boundaries[0])
        & (frame["target_date"] < boundaries[1])
    ]
    calibration = frame[
        (frame["target_date"] >= boundaries[1])
        & (frame["target_date"] < boundaries[2])
    ]
    threshold_validation = frame[
        frame["target_date"] >= boundaries[2]
    ]
    for name, split in (
        ("train", train),
        ("early_stop", early_stop),
        ("calibration", calibration),
        ("threshold", threshold_validation),
    ):
        if split.empty or split["signal_target"].nunique() < 2:
            raise ValueError(f"Некорректная временная выборка: {name}.")

    model = CatBoostClassifier(
        iterations=iterations,
        depth=depth,
        learning_rate=learning_rate,
        loss_function="Logloss",
        eval_metric="Logloss",
        random_seed=42,
        l2_leaf_reg=5.0,
        random_strength=0.5,
        verbose=50,
        allow_writing_files=False,
    )
    model.fit(
        train[FEATURE_COLUMNS],
        train["signal_target"],
        cat_features=CATEGORICAL_FEATURES,
        eval_set=(
            early_stop[FEATURE_COLUMNS],
            early_stop["signal_target"],
        ),
        early_stopping_rounds=50,
        use_best_model=True,
    )

    raw_calibration = model.predict_proba(
        calibration[FEATURE_COLUMNS]
    )[:, 1]
    calibrator = LogisticRegression(C=1e6)
    calibrator.fit(
        _logit(raw_calibration),
        calibration["signal_target"].to_numpy(),
    )
    coefficient = float(calibrator.coef_[0, 0])
    intercept = float(calibrator.intercept_[0])

    raw_threshold = model.predict_proba(
        threshold_validation[FEATURE_COLUMNS]
    )[:, 1]
    calibrated_threshold = calibrate_probabilities(
        raw_threshold, coefficient, intercept
    )
    actual_threshold = threshold_validation[
        "signal_target"
    ].to_numpy()
    (
        up_threshold,
        down_threshold,
        direction_threshold_scores,
    ) = select_direction_thresholds(
        calibrated_threshold,
        actual_threshold,
    )
    signal_mask = np.zeros(len(actual_threshold), dtype=bool)
    predicted = np.zeros(len(actual_threshold), dtype=int)
    if up_threshold is not None:
        up_mask = calibrated_threshold >= up_threshold
        signal_mask |= up_mask
        predicted[up_mask] = 1
    if down_threshold is not None:
        down_mask = calibrated_threshold <= 1.0 - down_threshold
        signal_mask |= down_mask
        predicted[down_mask] = 0
    validation_precision = (
        float(np.mean(predicted[signal_mask] == actual_threshold[signal_mask]))
        if np.any(signal_mask)
        else None
    )
    metrics = {
        "train_rows": int(len(train)),
        "early_stop_rows": int(len(early_stop)),
        "calibration_rows": int(len(calibration)),
        "threshold_validation_rows": int(len(threshold_validation)),
        "calibration_coefficient": coefficient,
        "calibration_intercept": intercept,
        "up_confidence_threshold": up_threshold,
        "down_confidence_threshold": down_threshold,
        "direction_threshold_scores": direction_threshold_scores,
        "validation_precision": validation_precision,
        "validation_coverage": float(signal_mask.mean()),
        "best_iteration": int(model.get_best_iteration()),
    }
    return model, metrics
