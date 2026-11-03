"""LightGBM и ансамбль для бинарного торгового сигнала."""

from lightgbm import early_stopping, LGBMClassifier, log_evaluation
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from signal_model import (
    _logit,
    calibrate_probabilities,
    select_direction_thresholds,
    train_binary_model,
)
from tabular_features import FEATURE_COLUMNS, NUMERIC_FEATURES


LIGHTGBM_FEATURES = ["ticker_code"] + NUMERIC_FEATURES


def build_ticker_mapping(frame: pd.DataFrame) -> dict[str, int]:
    return {
        ticker: index
        for index, ticker in enumerate(sorted(frame["ticker"].unique()))
    }


def lightgbm_frame(
    frame: pd.DataFrame,
    ticker_mapping: dict[str, int],
) -> pd.DataFrame:
    result = frame[NUMERIC_FEATURES].copy()
    result.insert(
        0,
        "ticker_code",
        frame["ticker"].map(ticker_mapping).fillna(-1).astype(int),
    )
    return result


def _temporal_splits(
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    dates = np.asarray(sorted(frame["target_date"].unique()))
    boundaries = [
        dates[int(len(dates) * fraction)]
        for fraction in (0.65, 0.75, 0.85)
    ]
    return (
        frame[frame["target_date"] < boundaries[0]],
        frame[
            (frame["target_date"] >= boundaries[0])
            & (frame["target_date"] < boundaries[1])
        ],
        frame[
            (frame["target_date"] >= boundaries[1])
            & (frame["target_date"] < boundaries[2])
        ],
        frame[frame["target_date"] >= boundaries[2]],
    )


def _calibration_metadata(
    raw_calibration: np.ndarray,
    calibration_target: np.ndarray,
    raw_threshold: np.ndarray,
    threshold_target: np.ndarray,
) -> tuple[np.ndarray, dict]:
    calibrator = LogisticRegression(C=1e6)
    calibrator.fit(_logit(raw_calibration), calibration_target)
    coefficient = float(calibrator.coef_[0, 0])
    intercept = float(calibrator.intercept_[0])
    calibrated_threshold = calibrate_probabilities(
        raw_threshold, coefficient, intercept
    )
    up_threshold, down_threshold, scores = select_direction_thresholds(
        calibrated_threshold, threshold_target
    )
    return calibrated_threshold, {
        "calibration_coefficient": coefficient,
        "calibration_intercept": intercept,
        "up_confidence_threshold": up_threshold,
        "down_confidence_threshold": down_threshold,
        "direction_threshold_scores": scores,
    }


def train_ensemble_models(
    frame: pd.DataFrame,
    iterations: int,
    depth: int,
    learning_rate: float,
) -> tuple[object, LGBMClassifier, dict]:
    catboost, catboost_metadata = train_binary_model(
        frame, iterations, depth, learning_rate
    )
    train, early_stop, calibration, threshold_validation = (
        _temporal_splits(frame)
    )
    ticker_mapping = build_ticker_mapping(train)
    lightgbm = LGBMClassifier(
        n_estimators=iterations,
        learning_rate=learning_rate,
        num_leaves=31,
        max_depth=-1,
        min_child_samples=50,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=5.0,
        random_state=42,
        n_jobs=-1,
        verbosity=-1,
    )
    lightgbm.fit(
        lightgbm_frame(train, ticker_mapping),
        train["signal_target"],
        eval_set=[
            (
                lightgbm_frame(early_stop, ticker_mapping),
                early_stop["signal_target"],
            )
        ],
        callbacks=[early_stopping(50, verbose=False), log_evaluation(0)],
        categorical_feature=["ticker_code"],
    )

    raw_lgb_calibration = lightgbm.predict_proba(
        lightgbm_frame(calibration, ticker_mapping)
    )[:, 1]
    raw_lgb_threshold = lightgbm.predict_proba(
        lightgbm_frame(threshold_validation, ticker_mapping)
    )[:, 1]
    calibrated_lgb_threshold, lightgbm_metadata = _calibration_metadata(
        raw_lgb_calibration,
        calibration["signal_target"].to_numpy(),
        raw_lgb_threshold,
        threshold_validation["signal_target"].to_numpy(),
    )

    raw_cat_threshold = catboost.predict_proba(
        threshold_validation[FEATURE_COLUMNS]
    )[:, 1]
    calibrated_cat_threshold = calibrate_probabilities(
        raw_cat_threshold,
        catboost_metadata["calibration_coefficient"],
        catboost_metadata["calibration_intercept"],
    )
    ensemble_threshold_probability = (
        calibrated_cat_threshold + calibrated_lgb_threshold
    ) / 2.0
    up_threshold, down_threshold, scores = select_direction_thresholds(
        ensemble_threshold_probability,
        threshold_validation["signal_target"].to_numpy(),
    )
    metadata = {
        "ticker_mapping": ticker_mapping,
        "lightgbm_features": LIGHTGBM_FEATURES,
        "catboost": catboost_metadata,
        "lightgbm": {
            **lightgbm_metadata,
            "best_iteration": int(lightgbm.best_iteration_),
        },
        "ensemble": {
            "catboost_weight": 0.5,
            "lightgbm_weight": 0.5,
            "up_confidence_threshold": up_threshold,
            "down_confidence_threshold": down_threshold,
            "direction_threshold_scores": scores,
        },
    }
    return catboost, lightgbm, metadata
