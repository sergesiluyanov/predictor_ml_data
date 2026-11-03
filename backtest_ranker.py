"""Nested walk-forward для cross-sectional long ranker."""

import argparse
from datetime import datetime
import json
from pathlib import Path

import numpy as np
import pandas as pd

from data_loader import load_index_history, load_ticker_histories
from ensemble_signal_model import lightgbm_frame
from evaluation_metrics import bootstrap_interval
from ranking_model import (
    build_ranking_frame,
    ranking_policy_metrics,
    select_long_candidates,
    train_ranker,
)
from train_direction_model import parse_tickers


def build_horizon_dataset(
    histories: dict[str, list[dict]],
    market_rows: list[dict],
    horizon: int,
) -> tuple[pd.DataFrame, list[dict]]:
    frames = []
    skipped = []
    for ticker, rows in histories.items():
        if not rows:
            continue
        try:
            frame = build_ranking_frame(
                rows, ticker, market_rows, horizon
            )
        except Exception as exc:
            skipped.append({"ticker": ticker, "reason": str(exc)})
            continue
        if len(frame) < 100:
            skipped.append({"ticker": ticker, "reason": "мало данных"})
            continue
        frames.append(frame)
    if not frames:
        raise ValueError(f"Нет данных для горизонта {horizon}.")
    dataset = pd.concat(frames, ignore_index=True)
    return (
        dataset.sort_values(["entry_date", "ticker"]).reset_index(drop=True),
        skipped,
    )


def run_horizon_walk_forward(
    dataset: pd.DataFrame,
    horizon: int,
    folds: int,
    test_days: int,
    min_train_days: int,
    iterations: int,
    learning_rate: float,
    roundtrip_cost_bps: float,
) -> pd.DataFrame:
    dates = np.asarray(sorted(dataset["entry_date"].unique()))
    first_test_index = max(min_train_days, len(dates) - folds * test_days)
    results = []
    for fold, start in enumerate(
        range(first_test_index, len(dates), test_days), start=1
    ):
        if fold > folds:
            break
        end = min(start + test_days, len(dates))
        train = dataset[dataset["entry_date"] < dates[start]].copy()
        test = dataset[
            (dataset["entry_date"] >= dates[start])
            & (dataset["entry_date"] <= dates[end - 1])
        ].copy()
        print(
            f"H{horizon} fold {fold}: train={len(train)}, "
            f"test={len(test)}, {dates[start]}..{dates[end - 1]}",
            flush=True,
        )
        model, metadata = train_ranker(
            train, iterations, learning_rate, roundtrip_cost_bps
        )
        scores = model.predict(
            lightgbm_frame(test, metadata["ticker_mapping"])
        )
        selected = select_long_candidates(
            test, scores, metadata["selected_top_fraction"]
        )
        result = test[
            [
                "ticker",
                "entry_date",
                "exit_date",
                "horizon_return",
                "market_return_20",
                "market_volatility_20",
                "log_turnover",
            ]
        ].copy()
        result["horizon"] = horizon
        result["fold"] = fold
        result["score"] = scores
        result["selected_top_fraction"] = metadata[
            "selected_top_fraction"
        ]
        result["selected"] = selected
        results.append(result)
    return pd.concat(results, ignore_index=True)


def ranker_metrics(
    frame: pd.DataFrame,
    roundtrip_cost_bps: float,
) -> dict:
    selected = frame["selected"].to_numpy(dtype=bool)
    base = ranking_policy_metrics(
        frame, selected, roundtrip_cost_bps
    )
    selected_net = (
        frame.loc[selected, "horizon_return"].to_numpy(dtype=float)
        - roundtrip_cost_bps / 10_000.0
    )
    selected_frame = frame.loc[selected, ["entry_date"]].copy()
    selected_frame["net_return"] = selected_net
    horizon = int(frame["horizon"].iloc[0])
    cohort = selected_frame.groupby("entry_date")["net_return"].mean()
    daily = cohort / horizon
    base["precision_95_ci"] = bootstrap_interval(
        (selected_net > 0).astype(float)
    )
    # Сигналы одного дня коррелированы, поэтому bootstrap выполняется
    # по дневным когортам, а не по отдельным бумагам.
    base["mean_signal_return_95_ci"] = bootstrap_interval(
        cohort.to_numpy()
    )
    base["median_daily_return_95_ci"] = bootstrap_interval(
        daily.to_numpy(), statistic=np.median
    )
    base["folds"] = {
        str(fold): ranking_policy_metrics(
            part,
            part["selected"].to_numpy(dtype=bool),
            roundtrip_cost_bps,
        )
        for fold, part in frame.groupby("fold")
    }
    return base


def ranker_gate(metrics: dict) -> dict:
    mean_return_ci = metrics.get("mean_signal_return_95_ci")
    median_daily_ci = metrics.get("median_daily_return_95_ci")
    folds = list(metrics.get("folds", {}).values())
    active_folds = [
        item for item in folds if item.get("signals", 0) > 0
    ]
    checks = {
        "at_least_200_signals": metrics.get("signals", 0) >= 200,
        "mean_signal_return_ci_above_zero": (
            mean_return_ci is not None and mean_return_ci[0] > 0.0
        ),
        "median_daily_return_ci_above_zero": (
            median_daily_ci is not None and median_daily_ci[0] > 0.0
        ),
        "profitable_date_share_at_least_55_percent": (
            metrics.get("profitable_date_share") is not None
            and metrics["profitable_date_share"] >= 0.55
        ),
        "signals_in_at_least_two_folds": len(active_folds) >= 2,
        "positive_median_in_every_active_fold": bool(active_folds)
        and all(
            item.get("median_daily_return") is not None
            and item["median_daily_return"] > 0
            for item in active_folds
        ),
        "max_drawdown_below_20_percent": (
            metrics.get("max_drawdown") is not None
            and metrics["max_drawdown"] > -0.20
        ),
    }
    return {"passed": all(checks.values()), "checks": checks}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Cross-sectional LGBMRanker 1/3/5 дней"
    )
    parser.add_argument("--tickers", default="all")
    parser.add_argument("--max-tickers", type=int, default=0)
    parser.add_argument("--days", type=int, default=1500)
    parser.add_argument("--horizons", default="1,3,5")
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--test-days", type=int, default=60)
    parser.add_argument("--min-train-days", type=int, default=252)
    parser.add_argument("--commission-bps", type=float, default=10.0)
    parser.add_argument("--slippage-bps", type=float, default=5.0)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--cache-dir", default="data/cache")
    parser.add_argument("--out-dir", default="reports")
    args = parser.parse_args()

    tickers = parse_tickers(args.tickers, args.max_tickers)
    market_rows = load_index_history(args.days, args.cache_dir)
    histories, load_errors = load_ticker_histories(
        tickers, args.days, args.workers, args.cache_dir
    )
    horizons = [
        int(value.strip())
        for value in args.horizons.split(",")
        if value.strip()
    ]
    roundtrip_cost_bps = 2.0 * (
        args.commission_bps + args.slippage_bps
    )
    all_predictions = []
    report = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "config": vars(args),
        "roundtrip_cost_bps": roundtrip_cost_bps,
        "load_errors": load_errors,
        "horizons": {},
    }
    for horizon in horizons:
        dataset, skipped = build_horizon_dataset(
            histories, market_rows, horizon
        )
        predictions = run_horizon_walk_forward(
            dataset,
            horizon,
            args.folds,
            args.test_days,
            args.min_train_days,
            args.iterations,
            args.learning_rate,
            roundtrip_cost_bps,
        )
        metrics = ranker_metrics(predictions, roundtrip_cost_bps)
        report["horizons"][str(horizon)] = {
            "tickers_tested": int(predictions["ticker"].nunique()),
            "skipped": skipped,
            "metrics": metrics,
            "quality_gate": ranker_gate(metrics),
        }
        all_predictions.append(predictions)
        print(
            f"H{horizon}: precision={metrics['precision']}, "
            f"median daily={metrics['median_daily_return']}, "
            f"gate={report['horizons'][str(horizon)]['quality_gate']['passed']}"
        )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"ranker_backtest_{timestamp}.json"
    csv_path = output_dir / f"ranker_backtest_{timestamp}.csv"
    json_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    pd.concat(all_predictions, ignore_index=True).to_csv(
        csv_path, index=False
    )
    print(f"Отчёт: {json_path}")
    print(f"Данные: {csv_path}")


if __name__ == "__main__":
    main()
