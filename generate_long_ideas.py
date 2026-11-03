"""Формирует ежедневный список long-кандидатов из cross-sectional ranker."""

import argparse
from datetime import datetime
import json
from pathlib import Path

from lightgbm import Booster
import pandas as pd

from data_loader import load_index_history, load_ticker_histories
from ensemble_signal_model import lightgbm_frame
from ranking_model import select_long_candidates
from tabular_features import FEATURE_COLUMNS, build_feature_frame
from train_direction_model import parse_tickers


def latest_universe_frame(
    histories: dict[str, list[dict]],
    market_rows: list[dict],
) -> pd.DataFrame:
    as_of = market_rows[-1]["tradedate"]
    rows = []
    for ticker, history in histories.items():
        if not history:
            continue
        frame = build_feature_frame(history, ticker, market_rows)
        if frame.empty or "_max_abs_return_20" not in frame:
            continue
        valid = frame.dropna(subset=FEATURE_COLUMNS)
        valid = valid[valid["_max_abs_return_20"] <= 0.25]
        if valid.empty:
            continue
        latest = valid.iloc[-1]
        if latest["tradedate"] != as_of:
            continue
        rows.append(latest)
    if not rows:
        raise ValueError("Нет актуальных бумаг для ранжирования.")
    result = pd.DataFrame(rows).reset_index(drop=True)
    result["entry_date"] = as_of
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ежедневные H5 long-кандидаты MOEX"
    )
    parser.add_argument("--tickers", default="all")
    parser.add_argument("--max-tickers", type=int, default=0)
    parser.add_argument("--days", type=int, default=180)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--cache-dir", default="data/cache")
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument(
        "--model", default="models/stock_long_ranker.txt"
    )
    parser.add_argument(
        "--out", default="models/latest_long_ideas.json"
    )
    args = parser.parse_args()

    model_path = Path(args.model)
    metadata_path = model_path.with_suffix(".json")
    if not model_path.is_file() or not metadata_path.is_file():
        raise SystemExit(
            "Long ranker не найден. Сначала запустите train_ranker_model.py."
        )
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    tickers = parse_tickers(args.tickers, args.max_tickers)
    market_rows = load_index_history(
        args.days, args.cache_dir, args.refresh
    )
    histories, load_errors = load_ticker_histories(
        tickers,
        args.days,
        args.workers,
        args.cache_dir,
        args.refresh,
    )
    frame = latest_universe_frame(histories, market_rows)
    frame = frame[
        frame["ticker"].isin(metadata["ticker_mapping"])
    ].reset_index(drop=True)
    if frame.empty:
        raise SystemExit("Нет актуальных тикеров из обучающей выборки.")
    model = Booster(model_file=str(model_path))
    scores = model.predict(
        lightgbm_frame(frame, metadata["ticker_mapping"])
    )
    selected = select_long_candidates(
        frame,
        scores,
        float(metadata["top_fraction"]),
        float(metadata["max_market_volatility_20"]),
    )
    candidates = frame.loc[
        selected, ["ticker", "log_turnover"]
    ].copy()
    candidates["score"] = scores[selected]
    candidates = candidates.sort_values("score", ascending=False)
    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "as_of": frame["entry_date"].iloc[0],
        "horizon_sessions": int(metadata["horizon"]),
        "universe_size": int(len(frame)),
        "market_return_20": float(frame["market_return_20"].iloc[0]),
        "market_volatility_20": float(
            frame["market_volatility_20"].iloc[0]
        ),
        "candidates": [
            {
                "ticker": row.ticker,
                "rank": rank,
                "score": float(row.score),
            }
            for rank, row in enumerate(
                candidates.itertuples(index=False), start=1
            )
        ],
        "load_errors": load_errors,
        "notice": (
            "Исторический quality gate пройден; live-валидация продолжается. "
            "Не является инвестиционной рекомендацией."
        ),
    }
    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(
        f"Сохранено: {output_path}; "
        f"кандидатов: {len(payload['candidates'])}"
    )


if __name__ == "__main__":
    main()
