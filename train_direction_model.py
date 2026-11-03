"""Обучение общей CatBoost-модели направления акций MOEX."""

import argparse
from datetime import datetime
import json
from pathlib import Path

from catboost import CatBoostClassifier
import numpy as np
import pandas as pd

from moex_client import (
    DEFAULT_LIQUID_TICKERS,
    fetch_index_last_n_days,
    fetch_last_n_days,
    fetch_liquid_tickers_full,
)
from tabular_features import (
    CATEGORICAL_FEATURES,
    CLASS_NAMES,
    FEATURE_COLUMNS,
    build_training_frame,
)


def parse_tickers(value: str, max_tickers: int) -> list[str]:
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


def train_model(
    frame: pd.DataFrame,
    iterations: int,
    depth: int,
    learning_rate: float,
) -> tuple[CatBoostClassifier, dict]:
    unique_dates = sorted(frame["target_date"].unique())
    if len(unique_dates) < 100:
        raise ValueError("Недостаточно дат для временного train/validation split.")
    cutoff = unique_dates[int(len(unique_dates) * 0.8)]
    train = frame[frame["target_date"] < cutoff].copy()
    validation = frame[frame["target_date"] >= cutoff].copy()

    model = CatBoostClassifier(
        iterations=iterations,
        depth=depth,
        learning_rate=learning_rate,
        loss_function="MultiClass",
        eval_metric="MultiClass",
        auto_class_weights="Balanced",
        random_seed=42,
        l2_leaf_reg=5.0,
        random_strength=0.5,
        verbose=50,
        allow_writing_files=False,
    )
    model.fit(
        train[FEATURE_COLUMNS],
        train["target_class"].astype(int),
        cat_features=CATEGORICAL_FEATURES,
        eval_set=(
            validation[FEATURE_COLUMNS],
            validation["target_class"].astype(int),
        ),
        early_stopping_rounds=50,
        use_best_model=True,
    )

    prediction = model.predict(validation[FEATURE_COLUMNS]).reshape(-1).astype(int)
    actual = validation["target_class"].to_numpy(dtype=int)
    accuracy = float(np.mean(prediction == actual))
    per_class_accuracy = {}
    for class_id, class_name in CLASS_NAMES.items():
        mask = actual == class_id
        per_class_accuracy[class_name] = (
            float(np.mean(prediction[mask] == actual[mask]))
            if np.any(mask)
            else None
        )
    class_returns = {
        str(class_id): float(
            train.loc[
                train["target_class"].astype(int) == class_id,
                "target_return",
            ].mean()
        )
        for class_id in CLASS_NAMES
    }
    metrics = {
        "train_rows": int(len(train)),
        "validation_rows": int(len(validation)),
        "validation_start": str(cutoff),
        "validation_accuracy": accuracy,
        "validation_per_class_accuracy": per_class_accuracy,
        "class_mean_returns": class_returns,
        "best_iteration": int(model.get_best_iteration()),
    }
    return model, metrics


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Обучение CatBoost-классификатора down/flat/up"
    )
    parser.add_argument("--tickers", default="all")
    parser.add_argument("--max-tickers", type=int, default=50)
    parser.add_argument("--days", type=int, default=1500)
    parser.add_argument("--threshold", type=float, default=0.003)
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument(
        "--out", default="models/stock_direction.cbm"
    )
    args = parser.parse_args()

    tickers = parse_tickers(args.tickers, args.max_tickers)
    print("Загрузка IMOEX...")
    market_rows = fetch_index_last_n_days("IMOEX", args.days)
    frames = []
    skipped = []
    for index, ticker in enumerate(tickers, start=1):
        print(f"[{index}/{len(tickers)}] {ticker}")
        try:
            rows = fetch_last_n_days(ticker, args.days)
            frame = build_training_frame(
                rows, ticker, market_rows, args.threshold
            )
        except Exception as exc:
            skipped.append({"ticker": ticker, "reason": str(exc)})
            continue
        if len(frame) < 100:
            skipped.append({"ticker": ticker, "reason": "мало данных"})
            continue
        frames.append(frame)

    if not frames:
        raise SystemExit(f"Нет данных для обучения. Пропущено: {skipped}")
    dataset = pd.concat(frames, ignore_index=True)
    dataset = dataset.sort_values(["target_date", "ticker"]).reset_index(drop=True)
    model, metrics = train_model(
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
        "model_type": "catboost_direction_3class",
        "model_path": str(output_path),
        "threshold": args.threshold,
        "feature_columns": FEATURE_COLUMNS,
        "categorical_features": CATEGORICAL_FEATURES,
        "class_names": {str(key): value for key, value in CLASS_NAMES.items()},
        "tickers": sorted(dataset["ticker"].unique().tolist()),
        "first_target_date": str(dataset["target_date"].min()),
        "last_target_date": str(dataset["target_date"].max()),
        "skipped": skipped,
        **metrics,
    }
    metadata_path = output_path.with_suffix(".json")
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(
        f"Validation accuracy: {metrics['validation_accuracy']:.2%}; "
        f"best iteration: {metrics['best_iteration']}"
    )
    print(f"Модель: {output_path}")
    print(f"Метаданные: {metadata_path}")


if __name__ == "__main__":
    main()
