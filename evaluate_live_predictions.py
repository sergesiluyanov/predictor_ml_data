"""Оценивает созревшие immutable live-прогнозы и строит сводку."""

import argparse
from datetime import datetime
import json
from pathlib import Path

from data_loader import load_index_history, load_ticker_histories
from live_validation import (
    evaluate_live_prediction,
    file_sha256,
    summarize_live_results,
    write_immutable_json,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Оценка live-прогнозов после завершения H5"
    )
    parser.add_argument(
        "--predictions-dir", default="data/live_predictions"
    )
    parser.add_argument("--results-dir", default="data/live_results")
    parser.add_argument(
        "--summary-out", default="reports/live_validation_summary.json"
    )
    parser.add_argument("--days", type=int, default=1500)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--cache-dir", default="data/cache")
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()

    prediction_paths = sorted(Path(args.predictions_dir).glob("*.json"))
    predictions = [
        (
            path,
            json.loads(path.read_text(encoding="utf-8")),
        )
        for path in prediction_paths
    ]
    pending = [
        (path, item)
        for path, item in predictions
        if not (Path(args.results_dir) / path.name).is_file()
    ]
    tickers = sorted(
        {
            candidate["ticker"]
            for _, item in pending
            for candidate in item.get("candidates", [])
        }
    )
    market_rows = load_index_history(
        args.days, args.cache_dir, args.refresh
    )
    if tickers:
        histories, load_errors = load_ticker_histories(
            tickers,
            args.days,
            args.workers,
            args.cache_dir,
            args.refresh,
        )
    else:
        histories, load_errors = {}, []

    created = 0
    still_pending = 0
    for prediction_path, prediction in pending:
        result = evaluate_live_prediction(
            prediction,
            histories,
            market_rows,
            float(prediction["roundtrip_cost_bps"]),
        )
        if result is None:
            still_pending += 1
            continue
        result["prediction_file"] = str(prediction_path)
        result["prediction_sha256"] = file_sha256(prediction_path)
        result["load_errors"] = load_errors
        result_path = Path(args.results_dir) / prediction_path.name
        if write_immutable_json(result_path, result):
            created += 1

    result_paths = sorted(Path(args.results_dir).glob("*.json"))
    results = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in result_paths
    ]
    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "method": (
            "next ticker open to fifth ticker close, "
            "including frozen round-trip costs"
        ),
        **summarize_live_results(results),
        "pending_predictions": still_pending,
    }
    summary_path = Path(args.summary_out)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(
        f"Новых результатов: {created}; ожидают: {still_pending}; "
        f"live gate: {summary['live_quality_gate']['passed']}"
    )
    print(f"Сводка: {summary_path}")


if __name__ == "__main__":
    main()
