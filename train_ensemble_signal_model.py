"""Обучение CatBoost и LightGBM на полном наборе сигналов."""

import argparse
from datetime import datetime
import json
from pathlib import Path

from ensemble_signal_model import train_ensemble_models
from signal_model import prepare_binary_dataset
from tabular_features import FEATURE_COLUMNS
from train_binary_signal_model import load_signal_source_dataset
from train_direction_model import parse_tickers


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Обучение CatBoost + LightGBM для торгового сигнала"
    )
    parser.add_argument("--tickers", default="all")
    parser.add_argument("--max-tickers", type=int, default=0)
    parser.add_argument("--days", type=int, default=1500)
    parser.add_argument("--min-abs-return", type=float, default=0.003)
    parser.add_argument("--max-abs-return", type=float, default=0.25)
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--cache-dir", default="data/cache")
    parser.add_argument("--catboost-out", default="models/stock_signal.cbm")
    parser.add_argument(
        "--lightgbm-out", default="models/stock_signal_lgbm.txt"
    )
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
    dataset = prepare_binary_dataset(source, args.min_abs_return)
    catboost, lightgbm, metadata = train_ensemble_models(
        dataset,
        args.iterations,
        args.depth,
        args.learning_rate,
    )

    catboost_path = Path(args.catboost_out)
    lightgbm_path = Path(args.lightgbm_out)
    catboost_path.parent.mkdir(parents=True, exist_ok=True)
    lightgbm_path.parent.mkdir(parents=True, exist_ok=True)
    catboost.save_model(catboost_path)
    lightgbm.booster_.save_model(lightgbm_path)

    common = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "min_abs_return": args.min_abs_return,
        "max_abs_return": args.max_abs_return,
        "feature_columns": FEATURE_COLUMNS,
        "tickers": sorted(dataset["ticker"].unique().tolist()),
        "dataset_rows": int(len(dataset)),
        "skipped": skipped,
    }
    catboost_metadata = {
        **common,
        "model_type": "catboost_binary_open_close_signal",
        "model_path": str(catboost_path),
        **metadata["catboost"],
    }
    lightgbm_metadata = {
        **common,
        "model_type": "lightgbm_binary_open_close_signal",
        "model_path": str(lightgbm_path),
        "ticker_mapping": metadata["ticker_mapping"],
        "lightgbm_features": metadata["lightgbm_features"],
        **metadata["lightgbm"],
    }
    ensemble_metadata = {
        **common,
        "model_type": "catboost_lightgbm_ensemble",
        "catboost_model_path": str(catboost_path),
        "lightgbm_model_path": str(lightgbm_path),
        **metadata["ensemble"],
    }
    catboost_path.with_suffix(".json").write_text(
        json.dumps(catboost_metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    lightgbm_path.with_suffix(".json").write_text(
        json.dumps(lightgbm_metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    Path("models/stock_signal_ensemble.json").write_text(
        json.dumps(ensemble_metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(
        "LightGBM thresholds: "
        f"up={lightgbm_metadata['up_confidence_threshold']}, "
        f"down={lightgbm_metadata['down_confidence_threshold']}"
    )
    print(f"CatBoost: {catboost_path}")
    print(f"LightGBM: {lightgbm_path}")


if __name__ == "__main__":
    main()
