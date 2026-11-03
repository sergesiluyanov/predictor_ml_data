"""Walk-forward-бэктест бинарной модели с зоной отсутствия сигнала."""

import argparse
from datetime import datetime
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from signal_model import (
    calibrate_probabilities,
    prepare_binary_dataset,
    train_binary_model,
)
from train_binary_signal_model import load_signal_source_dataset
from train_direction_model import parse_tickers
from tabular_features import FEATURE_COLUMNS


def signal_metrics(
    frame: pd.DataFrame,
    min_abs_return: float,
) -> dict:
    signal_mask = frame["signal"] != "none"
    actual_up = frame["strategy_return"] > 0
    predicted_up = frame["signal"] == "up"
    correct = predicted_up == actual_up
    significant = frame["strategy_return"].abs() >= min_abs_return
    up_mask = frame["signal"] == "up"
    down_mask = frame["signal"] == "down"

    def accuracy(mask: pd.Series) -> float | None:
        return float(correct[mask].mean()) if int(mask.sum()) else None

    return {
        "count": int(len(frame)),
        "signals": int(signal_mask.sum()),
        "coverage": float(signal_mask.mean()),
        "precision": accuracy(signal_mask),
        "significant_precision": accuracy(signal_mask & significant),
        "up_signals": int(up_mask.sum()),
        "up_precision": accuracy(up_mask),
        "down_signals": int(down_mask.sum()),
        "down_precision": accuracy(down_mask),
    }


def trading_metrics(
    frame: pd.DataFrame,
    position_column: str,
    commission_bps: float,
) -> dict:
    ticker_returns = []
    for _, ticker_frame in frame.groupby("ticker"):
        ticker_frame = ticker_frame.sort_values("target_date")
        position = ticker_frame[position_column].to_numpy(dtype=float)
        returns = ticker_frame["strategy_return"].to_numpy(dtype=float)
        turnover = np.abs(np.diff(position, prepend=0.0))
        net = position * returns - turnover * (commission_bps / 10_000.0)
        ticker_returns.append(float(np.prod(1.0 + net) - 1.0))
    values = np.asarray(ticker_returns)
    return {
        "mean_ticker_return": float(values.mean()),
        "median_ticker_return": float(np.median(values)),
        "profitable_tickers": int(np.sum(values > 0)),
        "ticker_count": int(len(values)),
    }


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
        model, metadata = train_binary_model(
            train,
            iterations=iterations,
            depth=depth,
            learning_rate=learning_rate,
        )
        raw_probability = model.predict_proba(test[FEATURE_COLUMNS])[:, 1]
        probability = calibrate_probabilities(
            raw_probability,
            metadata["calibration_coefficient"],
            metadata["calibration_intercept"],
        )
        up_threshold = metadata["up_confidence_threshold"]
        down_threshold = metadata["down_confidence_threshold"]
        signal = np.full(len(test), "none", dtype=object)
        if up_threshold is not None:
            signal[probability >= up_threshold] = "up"
        if down_threshold is not None:
            signal[probability <= 1.0 - down_threshold] = "down"

        result = test[
            [
                "ticker",
                "target_date",
                "current_price",
                "target_open",
                "target_price",
                "target_return",
                "strategy_return",
                "return_1",
            ]
        ].copy()
        result["fold"] = fold
        result["raw_up_probability"] = raw_probability
        result["up_probability"] = probability
        result["up_confidence_threshold"] = up_threshold
        result["down_confidence_threshold"] = down_threshold
        result["signal"] = signal
        result["model_position"] = (signal == "up").astype(int)
        result["momentum_position"] = (
            result["return_1"] > 0
        ).astype(int)
        results.append(result)
    return pd.concat(results, ignore_index=True)


def save_chart(
    frame: pd.DataFrame,
    output_path: Path,
    commission_bps: float,
) -> None:
    fig, ax = plt.subplots(figsize=(11, 6))
    for column, label in (
        ("model_position", "Binary CatBoost"),
        ("momentum_position", "Momentum"),
    ):
        curves = []
        for _, ticker_frame in frame.groupby("ticker"):
            ticker_frame = ticker_frame.sort_values("target_date")
            position = ticker_frame[column].to_numpy(dtype=float)
            returns = ticker_frame["strategy_return"].to_numpy(dtype=float)
            turnover = np.abs(np.diff(position, prepend=0.0))
            net = (
                position * returns
                - turnover * (commission_bps / 10_000.0)
            )
            curves.append(np.cumprod(1.0 + net) - 1.0)
        common = min(len(curve) for curve in curves)
        mean_curve = np.mean(
            np.stack([curve[:common] for curve in curves]), axis=0
        )
        ax.plot(mean_curve * 100, label=label)
    ax.set_title("Бинарный сигнал: средняя доходность по тикерам")
    ax.set_xlabel("Последовательные тестовые прогнозы")
    ax.set_ylabel("Доходность после комиссии, %")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=140)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Walk-forward бинарной модели торгового сигнала"
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
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--test-days", type=int, default=60)
    parser.add_argument("--min-train-days", type=int, default=252)
    parser.add_argument("--commission-bps", type=float, default=10.0)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--cache-dir", default="data/cache")
    parser.add_argument("--refresh-cache", action="store_true")
    parser.add_argument("--out-dir", default="reports")
    args = parser.parse_args()

    tickers = parse_tickers(args.tickers, args.max_tickers)
    source, skipped = load_signal_source_dataset(
        tickers=tickers,
        days=args.days,
        min_abs_return=args.min_abs_return,
        max_abs_return=args.max_abs_return,
        workers=args.workers,
        cache_dir=args.cache_dir,
        refresh_cache=args.refresh_cache,
    )
    predictions = run_walk_forward(
        source=source,
        folds=args.folds,
        test_days=args.test_days,
        min_train_days=args.min_train_days,
        min_abs_return=args.min_abs_return,
        iterations=args.iterations,
        depth=args.depth,
        learning_rate=args.learning_rate,
    )
    report = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "config": vars(args),
        "tickers_tested": sorted(predictions["ticker"].unique().tolist()),
        "skipped": skipped,
        "signal_metrics": signal_metrics(
            predictions, args.min_abs_return
        ),
        "model_trading": trading_metrics(
            predictions, "model_position", args.commission_bps
        ),
        "momentum_trading": trading_metrics(
            predictions, "momentum_position", args.commission_bps
        ),
    }

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"binary_backtest_{timestamp}.json"
    csv_path = output_dir / f"binary_backtest_{timestamp}.csv"
    chart_path = output_dir / f"binary_backtest_{timestamp}.png"
    json_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    predictions.to_csv(csv_path, index=False)
    save_chart(predictions, chart_path, args.commission_bps)

    metrics = report["signal_metrics"]
    trading = report["model_trading"]
    print(
        f"Precision={metrics['precision']:.2%}; "
        f"coverage={metrics['coverage']:.2%}; "
        f"signals={metrics['signals']}; "
        f"median return={trading['median_ticker_return']:.2%}"
    )
    print(f"Отчёт: {json_path}")
    print(f"Данные: {csv_path}")
    print(f"График: {chart_path}")


if __name__ == "__main__":
    main()
