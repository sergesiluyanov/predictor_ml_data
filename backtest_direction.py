"""Walk-forward-бэктест общей трёхклассовой CatBoost-модели."""

import argparse
import csv
from datetime import datetime
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from data_loader import load_index_history, load_ticker_histories
from tabular_features import (
    CLASS_NAMES,
    FEATURE_COLUMNS,
    build_training_frame,
    classify_returns,
)
from train_direction_model import parse_tickers, train_model


def classification_metrics(
    actual: np.ndarray, predicted: np.ndarray
) -> dict:
    recalls = {}
    for class_id, name in CLASS_NAMES.items():
        mask = actual == class_id
        recalls[name] = (
            float(np.mean(predicted[mask] == actual[mask]))
            if np.any(mask)
            else None
        )
    available_recalls = [
        value for value in recalls.values() if value is not None
    ]
    return {
        "count": int(len(actual)),
        "accuracy": float(np.mean(predicted == actual)),
        "balanced_accuracy": float(np.mean(available_recalls)),
        "recall": recalls,
    }


def trading_metrics(
    predictions: pd.DataFrame,
    position_column: str,
    commission_bps: float,
) -> dict:
    returns = []
    active_days = 0
    for _, ticker_frame in predictions.groupby("ticker"):
        ticker_frame = ticker_frame.sort_values("target_date")
        position = ticker_frame[position_column].to_numpy(dtype=float)
        actual = ticker_frame["strategy_return"].to_numpy(dtype=float)
        turnover = np.abs(np.diff(position, prepend=0.0))
        net = position * actual - turnover * (commission_bps / 10_000.0)
        returns.append(float(np.prod(1.0 + net) - 1.0))
        active_days += int(np.sum(position > 0))
    return {
        "mean_ticker_return": float(np.mean(returns)),
        "median_ticker_return": float(np.median(returns)),
        "profitable_tickers": int(np.sum(np.asarray(returns) > 0)),
        "ticker_count": len(returns),
        "active_days": active_days,
    }


def build_dataset(
    tickers: list[str],
    days: int,
    threshold: float,
    volatility_threshold_multiplier: float,
    max_abs_return: float,
    workers: int,
    cache_dir: str,
    refresh_cache: bool,
) -> tuple[pd.DataFrame, list[dict]]:
    market_rows = load_index_history(days, cache_dir, refresh_cache)
    histories, load_errors = load_ticker_histories(
        tickers, days, workers, cache_dir, refresh_cache
    )
    frames = []
    skipped = list(load_errors)
    for ticker in tickers:
        rows = histories.get(ticker, [])
        if not rows:
            if not any(item["ticker"] == ticker for item in skipped):
                skipped.append({"ticker": ticker, "reason": "нет данных"})
            continue
        try:
            frame = build_training_frame(
                rows,
                ticker,
                market_rows,
                threshold,
                volatility_threshold_multiplier,
                max_abs_return,
            )
        except Exception as exc:
            skipped.append({"ticker": ticker, "reason": str(exc)})
            continue
        if len(frame) < 100:
            skipped.append({"ticker": ticker, "reason": "мало данных"})
            continue
        frames.append(frame)
    if not frames:
        raise ValueError("Нет данных для бэктеста.")
    return (
        pd.concat(frames, ignore_index=True).sort_values(
            ["target_date", "ticker"]
        ),
        skipped,
    )


def run_walk_forward(
    dataset: pd.DataFrame,
    folds: int,
    test_days: int,
    min_train_days: int,
    iterations: int,
    depth: int,
    learning_rate: float,
    probability_threshold: float,
) -> pd.DataFrame:
    dates = np.asarray(sorted(dataset["target_date"].unique()))
    if len(dates) < min_train_days + test_days:
        raise ValueError("Недостаточно торговых дат для walk-forward.")
    first_test_index = max(min_train_days, len(dates) - folds * test_days)
    fold_frames = []

    for fold, start in enumerate(
        range(first_test_index, len(dates), test_days), start=1
    ):
        if fold > folds:
            break
        end = min(start + test_days, len(dates))
        train = dataset[dataset["target_date"] < dates[start]].copy()
        test = dataset[
            (dataset["target_date"] >= dates[start])
            & (dataset["target_date"] <= dates[end - 1])
        ].copy()
        print(
            f"Fold {fold}: train={len(train)}, test={len(test)}, "
            f"{dates[start]}..{dates[end - 1]}",
            flush=True,
        )
        model, training_metrics = train_model(
            train,
            iterations=iterations,
            depth=depth,
            learning_rate=learning_rate,
        )
        probabilities = model.predict_proba(test[FEATURE_COLUMNS])
        classes = np.asarray(model.classes_, dtype=int)
        class_to_column = {
            class_id: index for index, class_id in enumerate(classes)
        }
        for class_id in CLASS_NAMES:
            if class_id not in class_to_column:
                raise ValueError(f"В модели отсутствует класс {class_id}.")

        result = test[
            [
                "ticker",
                "target_date",
                "current_price",
                "target_price",
                "target_return",
                "strategy_return",
                "target_class",
                "target_threshold",
                "return_1",
            ]
        ].copy()
        result["fold"] = fold
        result["prob_down"] = probabilities[:, class_to_column[0]]
        result["prob_flat"] = probabilities[:, class_to_column[1]]
        result["prob_up"] = probabilities[:, class_to_column[2]]
        result["model_class"] = classes[np.argmax(probabilities, axis=1)]
        result["momentum_class"] = classify_returns(
            result["return_1"], result["target_threshold"]
        ).to_numpy()
        result["flat_class"] = 1
        fold_probability_threshold = (
            probability_threshold
            if probability_threshold > 0
            else training_metrics["recommended_probability_threshold"]
        )
        result["signal_probability_threshold"] = (
            fold_probability_threshold
        )
        result["model_position"] = (
            result["prob_up"] >= fold_probability_threshold
        ).astype(int)
        result["momentum_position"] = (
            result["return_1"] > result["target_threshold"]
        ).astype(int)
        result = result.rename(columns={"target_return": "actual_return"})
        fold_frames.append(result)
    return pd.concat(fold_frames, ignore_index=True)


def save_chart(
    predictions: pd.DataFrame,
    output_path: Path,
    commission_bps: float,
) -> None:
    fig, ax = plt.subplots(figsize=(11, 6))
    for position_column, label in (
        ("model_position", "CatBoost"),
        ("momentum_position", "Momentum"),
    ):
        curves = []
        for _, frame in predictions.groupby("ticker"):
            frame = frame.sort_values("target_date")
            position = frame[position_column].to_numpy(dtype=float)
            actual = frame["strategy_return"].to_numpy(dtype=float)
            turnover = np.abs(np.diff(position, prepend=0.0))
            net = (
                position * actual
                - turnover * (commission_bps / 10_000.0)
            )
            curves.append(np.cumprod(1.0 + net) - 1.0)
        common_length = min(len(curve) for curve in curves)
        mean_curve = np.mean(
            np.stack([curve[:common_length] for curve in curves]), axis=0
        )
        ax.plot(mean_curve * 100, label=label)
    ax.set_title("Средняя walk-forward long/cash-доходность по тикерам")
    ax.set_xlabel("Последовательные тестовые прогнозы")
    ax.set_ylabel("Доходность после комиссии, %")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=140)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Walk-forward CatBoost down/flat/up"
    )
    parser.add_argument(
        "--tickers",
        default="SBER,GAZP,LKOH,GMKN,NVTK,ROSN,TATN,MOEX,CHMF,IRAO",
    )
    parser.add_argument(
        "--max-tickers", type=int, default=0, help="0 — все тикеры"
    )
    parser.add_argument("--days", type=int, default=1500)
    parser.add_argument("--threshold", type=float, default=0.003)
    parser.add_argument(
        "--volatility-threshold-multiplier",
        type=float,
        default=0.0,
    )
    parser.add_argument("--max-abs-return", type=float, default=0.25)
    parser.add_argument(
        "--probability-threshold",
        type=float,
        default=-1.0,
        help="<=0 — выбирать на внутренней validation-выборке",
    )
    parser.add_argument("--iterations", type=int, default=300)
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--test-days", type=int, default=60)
    parser.add_argument("--min-train-days", type=int, default=252)
    parser.add_argument("--commission-bps", type=float, default=10.0)
    parser.add_argument("--out-dir", default="reports")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--cache-dir", default="data/cache")
    parser.add_argument("--refresh-cache", action="store_true")
    args = parser.parse_args()

    tickers = parse_tickers(args.tickers, args.max_tickers)
    dataset, skipped = build_dataset(
        tickers,
        args.days,
        args.threshold,
        args.volatility_threshold_multiplier,
        args.max_abs_return,
        args.workers,
        args.cache_dir,
        args.refresh_cache,
    )
    predictions = run_walk_forward(
        dataset=dataset,
        folds=args.folds,
        test_days=args.test_days,
        min_train_days=args.min_train_days,
        iterations=args.iterations,
        depth=args.depth,
        learning_rate=args.learning_rate,
        probability_threshold=args.probability_threshold,
    )

    actual = predictions["target_class"].to_numpy(dtype=int)
    report = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "config": vars(args),
        "tickers_tested": sorted(predictions["ticker"].unique().tolist()),
        "skipped": skipped,
        "model": {
            **classification_metrics(
                actual,
                predictions["model_class"].to_numpy(dtype=int),
            ),
            "trading": trading_metrics(
                predictions, "model_position", args.commission_bps
            ),
        },
        "momentum": {
            **classification_metrics(
                actual,
                predictions["momentum_class"].to_numpy(dtype=int),
            ),
            "trading": trading_metrics(
                predictions, "momentum_position", args.commission_bps
            ),
        },
        "always_flat": classification_metrics(
            actual, predictions["flat_class"].to_numpy(dtype=int)
        ),
    }

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"direction_backtest_{timestamp}.json"
    csv_path = output_dir / f"direction_backtest_{timestamp}.csv"
    chart_path = output_dir / f"direction_backtest_{timestamp}.png"
    json_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    predictions.to_csv(
        csv_path,
        index=False,
        quoting=csv.QUOTE_MINIMAL,
    )
    save_chart(predictions, chart_path, args.commission_bps)

    print(
        f"CatBoost accuracy={report['model']['accuracy']:.2%}, "
        f"balanced={report['model']['balanced_accuracy']:.2%}, "
        "mean long/cash="
        f"{report['model']['trading']['mean_ticker_return']:.2%}"
    )
    print(f"Отчёт: {json_path}")
    print(f"Данные: {csv_path}")
    print(f"График: {chart_path}")


if __name__ == "__main__":
    main()
