"""Cross-sectional LightGBM ranking для long-кандидатов."""

from lightgbm import early_stopping, LGBMRanker, log_evaluation
import numpy as np
import pandas as pd

from ensemble_signal_model import build_ticker_mapping, lightgbm_frame
from tabular_features import FEATURE_COLUMNS, build_feature_frame


TOP_FRACTIONS = (0.03, 0.05, 0.10)


def build_ranking_frame(
    rows: list[dict],
    ticker: str,
    market_rows: list[dict],
    horizon: int,
    max_abs_return: float = 0.50,
) -> pd.DataFrame:
    frame = build_feature_frame(rows, ticker, market_rows)
    if frame.empty:
        return frame
    frame["entry_date"] = frame["tradedate"].shift(-1)
    frame["exit_date"] = frame["tradedate"].shift(-horizon)
    frame["entry_open"] = frame["open"].shift(-1).astype(float)
    frame["exit_close"] = frame["close"].shift(-horizon).astype(float)
    frame["horizon_return"] = (
        frame["exit_close"] / frame["entry_open"] - 1.0
    )
    frame["horizon"] = horizon
    required = FEATURE_COLUMNS + [
        "entry_date",
        "exit_date",
        "horizon_return",
    ]
    valid = frame.dropna(subset=required)
    valid = valid[
        (valid["horizon_return"].abs() <= max_abs_return)
        & (valid["_max_abs_return_20"] <= 0.25)
    ]
    return valid.reset_index(drop=True)


def add_relevance_labels(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    percentile = result.groupby("entry_date")[
        "horizon_return"
    ].rank(pct=True, method="average")
    result["relevance"] = np.floor(percentile * 5).clip(0, 4).astype(int)
    return result


def _group_sizes(frame: pd.DataFrame) -> np.ndarray:
    return frame.groupby("entry_date", sort=False).size().to_numpy()


def select_long_candidates(
    frame: pd.DataFrame,
    scores: np.ndarray,
    top_fraction: float,
    max_market_volatility: float = 0.02,
) -> np.ndarray:
    """Top-N среди ликвидных бумаг в допустимом рыночном режиме."""
    ranked = frame.copy()
    ranked["_score"] = scores
    ranked["_selected"] = False
    ranked["_liquidity_pct"] = ranked.groupby("entry_date")[
        "log_turnover"
    ].rank(pct=True, method="average")
    eligible = ranked[
        (ranked["market_volatility_20"] < max_market_volatility)
        & (ranked["market_return_20"] >= 0.0)
        & (ranked["_liquidity_pct"] >= 1 / 3)
    ]
    for _, group in eligible.groupby("entry_date"):
        count = max(1, int(np.ceil(len(group) * top_fraction)))
        selected_index = group.nlargest(count, "_score").index
        ranked.loc[selected_index, "_selected"] = True
    return ranked["_selected"].to_numpy(dtype=bool)


def ranking_policy_metrics(
    frame: pd.DataFrame,
    selected: np.ndarray,
    roundtrip_cost_bps: float,
) -> dict:
    net_return = (
        frame["horizon_return"].to_numpy(dtype=float)
        - roundtrip_cost_bps / 10_000.0
    )
    selected_returns = net_return[selected]
    if not len(selected_returns):
        return {
            "signals": 0,
            "precision": None,
            "mean_signal_return": None,
            "median_signal_return": None,
            "mean_daily_return": None,
            "median_daily_return": None,
            "profitable_date_share": None,
            "cumulative_portfolio_return": None,
            "max_drawdown": None,
            "annualized_sharpe": None,
            "dates": 0,
        }
    selected_frame = frame.loc[selected, ["entry_date"]].copy()
    selected_frame["net_return"] = selected_returns
    horizon = (
        int(frame["horizon"].iloc[0]) if "horizon" in frame.columns else 1
    )
    cohort_returns = selected_frame.groupby("entry_date")[
        "net_return"
    ].mean() / horizon
    equity = np.cumprod(1.0 + cohort_returns.to_numpy())
    running_max = np.maximum.accumulate(equity)
    drawdowns = equity / running_max - 1.0
    standard_deviation = float(cohort_returns.std(ddof=1))
    return {
        "signals": int(len(selected_returns)),
        "precision": float(np.mean(selected_returns > 0)),
        "mean_signal_return": float(np.mean(selected_returns)),
        "median_signal_return": float(np.median(selected_returns)),
        "mean_daily_return": float(cohort_returns.mean()),
        "median_daily_return": float(cohort_returns.median()),
        "profitable_date_share": float(np.mean(cohort_returns > 0)),
        "cumulative_portfolio_return": float(equity[-1] - 1.0),
        "max_drawdown": float(drawdowns.min()),
        "annualized_sharpe": (
            float(cohort_returns.mean() / standard_deviation * np.sqrt(252))
            if standard_deviation > 0
            else None
        ),
        "dates": int(len(cohort_returns)),
    }


def train_ranker(
    frame: pd.DataFrame,
    iterations: int,
    learning_rate: float,
    roundtrip_cost_bps: float,
) -> tuple[LGBMRanker, dict]:
    dates = np.asarray(sorted(frame["entry_date"].unique()))
    cutoff = dates[int(len(dates) * 0.80)]
    train = add_relevance_labels(frame[frame["entry_date"] < cutoff])
    validation = add_relevance_labels(
        frame[frame["entry_date"] >= cutoff]
    )
    train = train.sort_values(["entry_date", "ticker"])
    validation = validation.sort_values(["entry_date", "ticker"])
    ticker_mapping = build_ticker_mapping(train)

    model = LGBMRanker(
        objective="lambdarank",
        metric="ndcg",
        n_estimators=iterations,
        learning_rate=learning_rate,
        num_leaves=31,
        min_child_samples=50,
        colsample_bytree=0.8,
        reg_lambda=5.0,
        random_state=42,
        n_jobs=-1,
        verbosity=-1,
    )
    model.fit(
        lightgbm_frame(train, ticker_mapping),
        train["relevance"],
        group=_group_sizes(train),
        eval_set=[
            (
                lightgbm_frame(validation, ticker_mapping),
                validation["relevance"],
            )
        ],
        eval_group=[_group_sizes(validation)],
        eval_at=[5, 10],
        callbacks=[early_stopping(50, verbose=False), log_evaluation(0)],
        categorical_feature=["ticker_code"],
    )
    validation_scores = model.predict(
        lightgbm_frame(validation, ticker_mapping)
    )
    policies = {}
    for fraction in TOP_FRACTIONS:
        selected = select_long_candidates(
            validation, validation_scores, fraction
        )
        policies[str(fraction)] = ranking_policy_metrics(
            validation, selected, roundtrip_cost_bps
        )
    eligible = {
        fraction: values
        for fraction, values in policies.items()
        if values["signals"] >= 100
        and values["mean_signal_return"] is not None
        and values["mean_signal_return"] > 0.0
        and values["median_daily_return"] is not None
        and values["median_daily_return"] > 0.0
    }
    candidates = eligible or {
        fraction: values
        for fraction, values in policies.items()
        if values["signals"] >= 100
    }
    selected_fraction = float(
        max(
            candidates,
            key=lambda fraction: (
                candidates[fraction]["median_daily_return"],
                candidates[fraction]["mean_signal_return"],
            ),
        )
    )
    metadata = {
        "ticker_mapping": ticker_mapping,
        "best_iteration": int(model.best_iteration_),
        "validation_policies": policies,
        "selected_top_fraction": selected_fraction,
    }
    return model, metadata
