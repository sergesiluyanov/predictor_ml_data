"""Финальное обучение пятидневного cross-sectional long ranker."""

import argparse
from datetime import datetime
import json
from pathlib import Path

from lightgbm import LGBMRanker, log_evaluation

from backtest_ranker import build_horizon_dataset
from data_loader import load_index_history, load_ticker_histories
from ensemble_signal_model import build_ticker_mapping, lightgbm_frame
from ranking_model import add_relevance_labels, _group_sizes, train_ranker
from tabular_features import NUMERIC_FEATURES
from train_direction_model import parse_tickers


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Обучение проверенного H5 LightGBM long ranker"
    )
    parser.add_argument("--tickers", default="all")
    parser.add_argument("--max-tickers", type=int, default=0)
    parser.add_argument("--days", type=int, default=1500)
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument(
        "--top-fraction",
        type=float,
        default=0.03,
        help="Зафиксированная nested walk-forward policy (по умолчанию 3%%)",
    )
    parser.add_argument("--commission-bps", type=float, default=10.0)
    parser.add_argument("--slippage-bps", type=float, default=5.0)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--cache-dir", default="data/cache")
    parser.add_argument("--out", default="models/stock_long_ranker.txt")
    args = parser.parse_args()

    if args.horizon != 5:
        raise SystemExit(
            "Разрешён только горизонт 5 дней: остальные quality gate не прошли."
        )
    if args.top_fraction != 0.03:
        raise SystemExit(
            "Разрешена только проверенная nested walk-forward policy top 3%."
        )

    tickers = parse_tickers(args.tickers, args.max_tickers)
    market_rows = load_index_history(args.days, args.cache_dir)
    histories, load_errors = load_ticker_histories(
        tickers, args.days, args.workers, args.cache_dir
    )
    dataset, skipped = build_horizon_dataset(
        histories, market_rows, args.horizon
    )
    roundtrip_cost_bps = 2.0 * (
        args.commission_bps + args.slippage_bps
    )

    _, validation_metadata = train_ranker(
        dataset,
        args.iterations,
        args.learning_rate,
        roundtrip_cost_bps,
    )
    full = add_relevance_labels(dataset).sort_values(
        ["entry_date", "ticker"]
    )
    ticker_mapping = build_ticker_mapping(full)
    best_iteration = max(1, validation_metadata["best_iteration"])
    model = LGBMRanker(
        objective="lambdarank",
        metric="ndcg",
        n_estimators=best_iteration,
        learning_rate=args.learning_rate,
        num_leaves=31,
        min_child_samples=50,
        colsample_bytree=0.8,
        reg_lambda=5.0,
        random_state=42,
        n_jobs=-1,
        verbosity=-1,
    )
    model.fit(
        lightgbm_frame(full, ticker_mapping),
        full["relevance"],
        group=_group_sizes(full),
        categorical_feature=["ticker_code"],
        callbacks=[log_evaluation(0)],
    )

    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model.booster_.save_model(output_path)
    metadata = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "model_type": "lightgbm_cross_sectional_long_ranker",
        "model_path": str(output_path),
        "horizon": args.horizon,
        "top_fraction": args.top_fraction,
        "max_market_volatility_20": 0.02,
        "min_market_return_20": 0.0,
        "min_liquidity_percentile": 1 / 3,
        "roundtrip_cost_bps": roundtrip_cost_bps,
        "best_iteration": best_iteration,
        "ticker_mapping": ticker_mapping,
        "features": ["ticker_code"] + NUMERIC_FEATURES,
        "dataset_rows": int(len(full)),
        "tickers": sorted(full["ticker"].unique().tolist()),
        "validation_policies": validation_metadata["validation_policies"],
        "load_errors": load_errors,
        "skipped": skipped,
        "quality_gate": "passed_historical_nested_walk_forward",
    }
    output_path.with_suffix(".json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(
        f"Сохранено: {output_path}; H{args.horizon}, "
        f"top {metadata['top_fraction']:.0%}, "
        f"{len(metadata['tickers'])} тикеров"
    )


if __name__ == "__main__":
    main()
