"""Расширенные метрики устойчивости торговых сигналов."""

from collections.abc import Callable

import numpy as np
import pandas as pd


def bootstrap_interval(
    values: np.ndarray,
    statistic: Callable[[np.ndarray], float] = np.mean,
    iterations: int = 1000,
    seed: int = 42,
) -> list[float] | None:
    values = np.asarray(values, dtype=float)
    if len(values) < 2:
        return None
    rng = np.random.default_rng(seed)
    samples = np.empty(iterations, dtype=float)
    for index in range(iterations):
        sample = rng.choice(values, size=len(values), replace=True)
        samples[index] = statistic(sample)
    return [
        float(np.percentile(samples, 2.5)),
        float(np.percentile(samples, 97.5)),
    ]


def probability_metrics(
    actual: np.ndarray,
    probability: np.ndarray,
    bins: int = 10,
) -> dict:
    actual = np.asarray(actual, dtype=float)
    probability = np.clip(
        np.asarray(probability, dtype=float), 1e-6, 1.0 - 1e-6
    )
    log_loss = -np.mean(
        actual * np.log(probability)
        + (1.0 - actual) * np.log(1.0 - probability)
    )
    brier = np.mean(np.square(probability - actual))
    edges = np.linspace(0.0, 1.0, bins + 1)
    expected_calibration_error = 0.0
    calibration_bins = []
    for index in range(bins):
        lower, upper = edges[index], edges[index + 1]
        mask = (probability >= lower) & (
            probability < upper if index < bins - 1 else probability <= upper
        )
        count = int(mask.sum())
        if not count:
            continue
        confidence = float(probability[mask].mean())
        frequency = float(actual[mask].mean())
        expected_calibration_error += (
            count / len(actual) * abs(confidence - frequency)
        )
        calibration_bins.append(
            {
                "from": float(lower),
                "to": float(upper),
                "count": count,
                "mean_probability": confidence,
                "actual_frequency": frequency,
            }
        )
    return {
        "log_loss": float(log_loss),
        "brier_score": float(brier),
        "expected_calibration_error": float(
            expected_calibration_error
        ),
        "calibration_bins": calibration_bins,
    }


def signal_quality_metrics(
    frame: pd.DataFrame,
    signal_column: str,
    probability_column: str,
    min_abs_return: float,
    include_bootstrap: bool = True,
) -> dict:
    signal = frame[signal_column]
    probability = frame[probability_column].to_numpy(dtype=float)
    actual_up = (frame["strategy_return"] > 0).to_numpy()
    predicted_up = (signal == "up").to_numpy()
    signal_mask = (signal != "none").to_numpy()
    significant = (
        frame["strategy_return"].abs().to_numpy() >= min_abs_return
    )
    correct = predicted_up == actual_up
    up_mask = (signal == "up").to_numpy()
    down_mask = (signal == "down").to_numpy()

    def ratio(mask: np.ndarray) -> float | None:
        return float(correct[mask].mean()) if np.any(mask) else None

    result = {
        "count": int(len(frame)),
        "signals": int(signal_mask.sum()),
        "coverage": float(signal_mask.mean()),
        "precision": ratio(signal_mask),
        "significant_precision": ratio(signal_mask & significant),
        "up_signals": int(up_mask.sum()),
        "up_precision": ratio(up_mask),
        "up_recall": (
            float(np.sum(up_mask & actual_up) / np.sum(actual_up))
            if np.any(actual_up)
            else None
        ),
        "down_signals": int(down_mask.sum()),
        "down_precision": ratio(down_mask),
        "down_recall": (
            float(np.sum(down_mask & ~actual_up) / np.sum(~actual_up))
            if np.any(~actual_up)
            else None
        ),
        **probability_metrics(actual_up.astype(int), probability),
    }
    if include_bootstrap and np.any(signal_mask):
        result["precision_95_ci"] = bootstrap_interval(
            correct[signal_mask].astype(float)
        )
        result["coverage_95_ci"] = bootstrap_interval(
            signal_mask.astype(float)
        )
    return result


def trading_quality_metrics(
    frame: pd.DataFrame,
    position_column: str,
    commission_bps: float,
    slippage_bps: float,
) -> dict:
    strategy_returns = []
    buy_hold_returns = []
    total_cost_bps = commission_bps + slippage_bps
    for _, ticker_frame in frame.groupby("ticker"):
        ticker_frame = ticker_frame.sort_values("target_date")
        position = ticker_frame[position_column].to_numpy(dtype=float)
        actual = ticker_frame["strategy_return"].to_numpy(dtype=float)
        turnover = np.abs(np.diff(position, prepend=0.0))
        net = position * actual - turnover * (total_cost_bps / 10_000.0)
        strategy_returns.append(float(np.prod(1.0 + net) - 1.0))
        close_returns = ticker_frame["target_return"].to_numpy(dtype=float)
        buy_hold_returns.append(
            float(np.prod(1.0 + close_returns) - 1.0)
        )
    strategy = np.asarray(strategy_returns)
    buy_hold = np.asarray(buy_hold_returns)
    return {
        "commission_bps": commission_bps,
        "slippage_bps": slippage_bps,
        "mean_ticker_return": float(strategy.mean()),
        "mean_return_95_ci": bootstrap_interval(strategy),
        "median_ticker_return": float(np.median(strategy)),
        "median_return_95_ci": bootstrap_interval(
            strategy, statistic=np.median
        ),
        "profitable_tickers": int(np.sum(strategy > 0)),
        "profitable_ticker_share": float(np.mean(strategy > 0)),
        "ticker_count": int(len(strategy)),
        "mean_buy_hold_return": float(buy_hold.mean()),
        "median_buy_hold_return": float(np.median(buy_hold)),
    }


def add_evaluation_groups(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["year"] = pd.to_datetime(result["target_date"]).dt.year
    market_return = result["market_return_20"]
    market_volatility = result["market_volatility_20"]
    result["market_regime"] = np.select(
        [
            market_volatility >= 0.02,
            market_return >= 0.03,
            market_return <= -0.03,
        ],
        ["high_volatility", "uptrend", "downtrend"],
        default="sideways",
    )
    ranked_turnover = result["log_turnover"].rank(
        method="first", pct=True
    )
    result["liquidity_group"] = pd.cut(
        ranked_turnover,
        bins=[0.0, 1 / 3, 2 / 3, 1.0],
        labels=["low", "medium", "high"],
        include_lowest=True,
    ).astype(str)
    return result


def grouped_signal_metrics(
    frame: pd.DataFrame,
    signal_column: str,
    probability_column: str,
    min_abs_return: float,
) -> dict:
    grouped = add_evaluation_groups(frame)
    result = {}
    for group_column in (
        "fold",
        "year",
        "market_regime",
        "liquidity_group",
    ):
        result[group_column] = {
            str(group): signal_quality_metrics(
                part,
                signal_column,
                probability_column,
                min_abs_return,
                include_bootstrap=False,
            )
            for group, part in grouped.groupby(group_column, observed=True)
        }
    return result


def quality_gate(
    signal_metrics: dict,
    trading_metrics: dict,
    groups: dict | None = None,
) -> dict:
    precision = signal_metrics.get("precision")
    precision_interval = signal_metrics.get("precision_95_ci")
    checks = {
        "precision_at_least_55_percent": (
            precision is not None and precision >= 0.55
        ),
        "precision_ci_lower_at_least_50_percent": (
            precision_interval is not None
            and precision_interval[0] >= 0.50
        ),
        "coverage_at_least_2_percent": (
            signal_metrics.get("coverage", 0.0) >= 0.02
        ),
        "positive_median_return": (
            trading_metrics.get("median_ticker_return", 0.0) > 0.0
        ),
        "profitable_ticker_share_at_least_50_percent": (
            trading_metrics.get("profitable_ticker_share", 0.0) >= 0.50
        ),
    }
    if groups:
        fold_values = list(groups.get("fold", {}).values())
        checks["signals_present_in_every_fold"] = bool(
            fold_values
        ) and all(value.get("signals", 0) > 0 for value in fold_values)
        for group_name in ("market_regime", "liquidity_group"):
            values = [
                value
                for value in groups.get(group_name, {}).values()
                if value.get("signals", 0) >= 100
            ]
            checks[f"{group_name}_precision_at_least_50_percent"] = bool(
                values
            ) and all(
                value.get("precision") is not None
                and value["precision"] >= 0.50
                for value in values
            )
    return {"passed": all(checks.values()), "checks": checks}
