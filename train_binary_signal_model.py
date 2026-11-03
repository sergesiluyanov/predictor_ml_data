"""Обучение бинарной модели next-open → next-close."""

import argparse
from datetime import datetime
import json
from pathlib import Path

import pandas as pd

from data_loader import load_index_history, load_ticker_histories
from signal_model import prepare_binary_dataset, train_binary_model
from tabular_features import (
    CATEGORICAL_FEATURES,
    FEATURE_COLUMNS,
    build_training_frame,
)
from train_direction_model import parse_tickers


def load_signal_source_dataset(
    tickers: list[str],
    days: int,
    min_abs_return: float,
    max_abs_return: float,
    workers: int,
    cache_dir: str,
    refresh_cache: bool,
) -> tuple[pd.DataFrame, list[dict]]:
    market_rows = load_index_history(days, cache_dir, refresh_cache)
    histories, errors = load_ticker_histories(
        tickers, days, workers, cache_dir, refresh_cache
    )
    frames = []
    skipped = list(errors)
    for ticker in tickers:
        rows = histories.get(ticker, [])
        if not rows:
            if not any(item["ticker"] == ticker for item in skipped):
                skipped.append({"ticker": ticker, "reason": "нет данных"})
            continue
        try:
            source = build_training_frame(
                rows=rows,
                ticker=ticker,
                market_rows=market_rows,
                threshold=0.003,
                volatility_threshold_multiplier=0.0,
                max_abs_return=max_abs_return,
            )
            signal_frame = prepare_binary_dataset(source, min_abs_return)
        except Exception as exc:
            skipped.append({"ticker": ticker, "reason": str(exc)})
            continue
        if len(signal_frame) < 100:
            skipped.append({"ticker": ticker, "reason": "мало сигналов"})
            continue
        frames.append(source)
    if not frames:
        raise ValueError("Нет данных для бинарной модели.")
    dataset = pd.concat(frames, ignore_index=True)
    return (
        dataset.sort_values(["target_date", "ticker"]).reset_index(drop=True),
        skipped,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Бинарная CatBoost-модель торгового сигнала"
    )
    parser.add_argument("--tickers", default="all")
    parser.add_argument(
        "--max-tickers", type=int, default=0, help="0 — все тикеры"
    )
    parser.add_argument("--days", type=int, default=1500)
    parser.add_argument("--min-abs-return", type=float, default=0.003)
    parser.add_argument("--max-abs-return", type=float, default=0.25)
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--cache-dir", default="data/cache")
    parser.add_argument("--refresh-cache", action="store_true")
    parser.add_argument("--out", default="models/stock_signal.cbm")
    args = parser.parse_args()

    tickers = parse_tickers(args.tickers, args.max_tickers)
    source_dataset, skipped = load_signal_source_dataset(
        tickers=tickers,
        days=args.days,
        min_abs_return=args.min_abs_return,
        max_abs_return=args.max_abs_return,
        workers=args.workers,
        cache_dir=args.cache_dir,
        refresh_cache=args.refresh_cache,
    )
    dataset = prepare_binary_dataset(
        source_dataset, args.min_abs_return
    )
    model, metrics = train_binary_model(
        dataset,
        iterations=args.iterations,
        depth=args.depth,
        learning_rate=args.learning_rate,
    )

    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(output_path)
    metadata = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "model_type": "catboost_binary_open_close_signal",
        "model_path": str(output_path),
        "min_abs_return": args.min_abs_return,
        "max_abs_return": args.max_abs_return,
        "feature_columns": FEATURE_COLUMNS,
        "categorical_features": CATEGORICAL_FEATURES,
        "tickers": sorted(dataset["ticker"].unique().tolist()),
        "first_target_date": str(dataset["target_date"].min()),
        "last_target_date": str(dataset["target_date"].max()),
        "dataset_rows": int(len(dataset)),
        "up_share": float(dataset["signal_target"].mean()),
        "skipped": skipped,
        **metrics,
    }
    metadata_path = output_path.with_suffix(".json")
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(
        f"Precision: {metrics['validation_precision']:.2%}; "
        f"coverage: {metrics['validation_coverage']:.2%}; "
        f"up threshold: {metrics['up_confidence_threshold']}; "
        f"down threshold: {metrics['down_confidence_threshold']}"
    )
    print(f"Модель: {output_path}")
    print(f"Метаданные: {metadata_path}")


if __name__ == "__main__":
    main()
