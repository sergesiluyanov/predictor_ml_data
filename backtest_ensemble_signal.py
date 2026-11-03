"""Walk-forward-сравнение CatBoost, LightGBM и ансамбля."""

import argparse
from datetime import datetime
import json
from pathlib import Path

import numpy as np
import pandas as pd

from backtest_binary_signal import signal_metrics, trading_metrics
from ensemble_signal_model import (
    lightgbm_frame,
    train_ensemble_models,
)
from signal_model import calibrate_probabilities, prepare_binary_dataset
from tabular_features import FEATURE_COLUMNS
from train_binary_signal_model import load_signal_source_dataset
from train_direction_model import parse_tickers


def _signals(
    probabilities: np.ndarray,
    metadata: dict,
) -> np.ndarray:
    result = np.full(len(probabilities), "none", dtype=object)
    up_threshold = metadata.get("up_confidence_threshold")
    down_threshold = metadata.get("down_confidence_threshold")
    if up_threshold is not None:
        result[probabilities >= float(up_threshold)] = "up"
    if down_threshold is not None:
        result[
            probabilities <= 1.0 - float(down_threshold)
        ] = "down"
    return result


def run_walk_forward(
    source: pd.DataFrame,
    folds: int,
    test_days: int,
    min_train_days: int,
    min_abs_return: float,
    iterations: int,
    depth: int,
    learning_rate: float,
) -> pd.DataFrame:
    dates = np.asarray(sorted(source["target_date"].unique()))
    first_test_index = max(min_train_days, len(dates) - folds * test_days)
    results = []

    for fold, start in enumerate(
        range(first_test_index, len(dates), test_days), start=1
    ):
        if fold > folds:
            break
        end = min(start + test_days, len(dates))
        train_source = source[source["target_date"] < dates[start]]
        train = prepare_binary_dataset(train_source, min_abs_return)
        test = source[
            (source["target_date"] >= dates[start])
            & (source["target_date"] <= dates[end - 1])
        ].copy()
        print(
            f"Fold {fold}: train={len(train)}, test={len(test)}, "
            f"{dates[start]}..{dates[end - 1]}",
            flush=True,
        )
        catboost, lightgbm, metadata = train_ensemble_models(
            train, iterations, depth, learning_rate
        )
        raw_cat = catboost.predict_proba(test[FEATURE_COLUMNS])[:, 1]
        cat_probability = calibrate_probabilities(
            raw_cat,
            metadata["catboost"]["calibration_coefficient"],
            metadata["catboost"]["calibration_intercept"],
        )
        raw_lgb = lightgbm.predict_proba(
            lightgbm_frame(test, metadata["ticker_mapping"])
        )[:, 1]
        lgb_probability = calibrate_probabilities(
            raw_lgb,
            metadata["lightgbm"]["calibration_coefficient"],
            metadata["lightgbm"]["calibration_intercept"],
        )
        ensemble_probability = (
            cat_probability * metadata["ensemble"]["catboost_weight"]
            + lgb_probability * metadata["ensemble"]["lightgbm_weight"]
        )

        result = test[
            [
                "ticker",
                "target_date",
                "target_return",
                "strategy_return",
            ]
        ].copy()
        result["fold"] = fold
        for name, probability, model_metadata in (
            ("catboost", cat_probability, metadata["catboost"]),
            ("lightgbm", lgb_probability, metadata["lightgbm"]),
            ("ensemble", ensemble_probability, metadata["ensemble"]),
        ):
            result[f"{name}_up_probability"] = probability
            result[f"{name}_signal"] = _signals(
                probability, model_metadata
            )
            result[f"{name}_position"] = (
                result[f"{name}_signal"] == "up"
            ).astype(int)
        results.append(result)
    return pd.concat(results, ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="CatBoost vs LightGBM vs ensemble"
    )
    parser.add_argument("--tickers", default="all")
    parser.add_argument("--max-tickers", type=int, default=0)
    parser.add_argument("--days", type=int, default=1500)
    parser.add_argument("--min-abs-return", type=float, default=0.003)
    parser.add_argument("--max-abs-return", type=float, default=0.25)
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--test-days", type=int, default=60)
    parser.add_argument("--min-train-days", type=int, default=252)
    parser.add_argument("--commission-bps", type=float, default=10.0)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--cache-dir", default="data/cache")
    parser.add_argument("--out-dir", default="reports")
    args = parser.parse_args()

    tickers = parse_tickers(args.tickers, args.max_tickers)
    source, skipped = load_signal_source_dataset(
        tickers,
        args.days,
        args.min_abs_return,
        args.max_abs_return,
        args.workers,
        args.cache_dir,
        False,
    )
    predictions = run_walk_forward(
        source,
        args.folds,
        args.test_days,
        args.min_train_days,
        args.min_abs_return,
        args.iterations,
        args.depth,
        args.learning_rate,
    )
    report = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "config": vars(args),
        "tickers_tested": sorted(predictions["ticker"].unique()),
        "skipped": skipped,
        "models": {},
    }
    for name in ("catboost", "lightgbm", "ensemble"):
        metric_frame = predictions.assign(
            signal=predictions[f"{name}_signal"]
        )
        report["models"][name] = {
            "signals": signal_metrics(
                metric_frame, args.min_abs_return
            ),
            "trading": trading_metrics(
                predictions,
                f"{name}_position",
                args.commission_bps,
            ),
        }

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"ensemble_backtest_{timestamp}.json"
    csv_path = output_dir / f"ensemble_backtest_{timestamp}.csv"
    json_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    predictions.to_csv(csv_path, index=False)
    for name, values in report["models"].items():
        print(
            f"{name}: precision={values['signals']['precision']}, "
            f"coverage={values['signals']['coverage']:.2%}, "
            "median return="
            f"{values['trading']['median_ticker_return']:.2%}"
        )
    print(f"Отчёт: {json_path}")
    print(f"Данные: {csv_path}")


if __name__ == "__main__":
    main()
