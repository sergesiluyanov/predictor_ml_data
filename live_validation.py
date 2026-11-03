"""Неизменяемые live-прогнозы и их оценка после истечения горизонта."""

from datetime import datetime
import hashlib
import json
from pathlib import Path

import numpy as np

from evaluation_metrics import bootstrap_interval


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_immutable_json(path: str | Path, payload: dict) -> bool:
    """Создаёт файл один раз; существующий прогноз никогда не меняется."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        with target.open("x", encoding="utf-8") as output:
            json.dump(payload, output, ensure_ascii=False, indent=2)
            output.write("\n")
    except FileExistsError:
        return False
    return True


def archive_live_prediction(
    payload: dict,
    model_path: str | Path,
    metadata_path: str | Path,
    directory: str | Path,
) -> tuple[Path, bool]:
    archived = {
        **payload,
        "model_sha256": file_sha256(model_path),
        "metadata_sha256": file_sha256(metadata_path),
        "recorded_at": datetime.now().isoformat(timespec="seconds"),
        "status": "pending",
    }
    path = Path(directory) / f"{payload['as_of']}.json"
    return path, write_immutable_json(path, archived)


def _rows_after(
    rows: list[dict], as_of: str, horizon: int
) -> list[dict]:
    ordered = sorted(rows, key=lambda item: item["tradedate"])
    return [
        item
        for item in ordered
        if item["tradedate"] > as_of
        and item.get("open") is not None
        and item.get("close") is not None
    ][:horizon]


def evaluate_live_prediction(
    prediction: dict,
    histories: dict[str, list[dict]],
    market_rows: list[dict],
    roundtrip_cost_bps: float,
) -> dict | None:
    """Возвращает результат только когда полностью созрели все H5 сигналы."""
    horizon = int(prediction["horizon_sessions"])
    as_of = str(prediction["as_of"])
    candidates = prediction.get("candidates", [])
    if not candidates:
        market_future = _rows_after(market_rows, as_of, horizon)
        if len(market_future) < horizon:
            return None
        return {
            "as_of": as_of,
            "evaluated_at": datetime.now().isoformat(timespec="seconds"),
            "horizon_sessions": horizon,
            "signals": 0,
            "cash": True,
            "entry_date": market_future[0]["tradedate"],
            "exit_date": market_future[-1]["tradedate"],
            "candidates": [],
            "mean_net_return": 0.0,
            "median_net_return": 0.0,
            "profitable_share": None,
        }

    evaluated = []
    for candidate in candidates:
        ticker = candidate["ticker"]
        future = _rows_after(histories.get(ticker, []), as_of, horizon)
        if len(future) < horizon:
            return None
        entry = future[0]
        exit_row = future[-1]
        gross_return = float(exit_row["close"]) / float(entry["open"]) - 1.0
        net_return = gross_return - roundtrip_cost_bps / 10_000.0
        evaluated.append(
            {
                **candidate,
                "entry_date": entry["tradedate"],
                "entry_open": float(entry["open"]),
                "exit_date": exit_row["tradedate"],
                "exit_close": float(exit_row["close"]),
                "gross_return": gross_return,
                "net_return": net_return,
                "profitable": net_return > 0.0,
            }
        )
    returns = np.asarray(
        [item["net_return"] for item in evaluated], dtype=float
    )
    return {
        "as_of": as_of,
        "evaluated_at": datetime.now().isoformat(timespec="seconds"),
        "horizon_sessions": horizon,
        "signals": len(evaluated),
        "cash": False,
        "candidates": evaluated,
        "mean_net_return": float(returns.mean()),
        "median_net_return": float(np.median(returns)),
        "profitable_share": float(np.mean(returns > 0.0)),
    }


def summarize_live_results(results: list[dict]) -> dict:
    completed = sorted(results, key=lambda item: item["as_of"])
    active = [item for item in completed if item["signals"] > 0]
    daily = np.asarray(
        [item["mean_net_return"] for item in active], dtype=float
    )
    signal_returns = np.asarray(
        [
            candidate["net_return"]
            for item in active
            for candidate in item["candidates"]
        ],
        dtype=float,
    )
    if len(daily):
        allocated_returns = np.asarray(
            [
                item["mean_net_return"]
                / max(1, int(item["horizon_sessions"]))
                for item in active
            ],
            dtype=float,
        )
        equity = np.cumprod(1.0 + allocated_returns)
        running_max = np.maximum.accumulate(equity)
        max_drawdown = float((equity / running_max - 1.0).min())
        daily_ci = bootstrap_interval(daily)
    else:
        max_drawdown = None
        daily_ci = None
    checks = {
        "at_least_30_independent_dates": len(completed) >= 30,
        "at_least_200_signals": len(signal_returns) >= 200,
        "mean_date_return_ci_above_zero": (
            daily_ci is not None and daily_ci[0] > 0.0
        ),
        "max_drawdown_below_20_percent": (
            max_drawdown is not None and max_drawdown > -0.20
        ),
    }
    return {
        "completed_dates": len(completed),
        "active_dates": len(active),
        "cash_dates": len(completed) - len(active),
        "signals": int(len(signal_returns)),
        "mean_signal_return": (
            float(signal_returns.mean()) if len(signal_returns) else None
        ),
        "signal_precision": (
            float(np.mean(signal_returns > 0.0))
            if len(signal_returns)
            else None
        ),
        "mean_date_return": float(daily.mean()) if len(daily) else None,
        "mean_date_return_95_ci": daily_ci,
        "max_drawdown": max_drawdown,
        "live_quality_gate": {
            "passed": all(checks.values()),
            "checks": checks,
        },
    }
