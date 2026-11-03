"""Причинные табличные признаки для модели направления цены."""

import numpy as np
import pandas as pd


NUMERIC_FEATURES = [
    "return_1",
    "return_3",
    "return_5",
    "return_10",
    "return_20",
    "mean_return_5",
    "mean_return_10",
    "volatility_5",
    "volatility_10",
    "volatility_20",
    "sma_gap_5",
    "sma_gap_10",
    "sma_gap_20",
    "rsi_14",
    "atr_14",
    "range_pct",
    "overnight_gap",
    "intraday_return",
    "close_position",
    "volume_change",
    "volume_z_20",
    "log_turnover",
    "turnover_change",
    "turnover_z_20",
    "zero_volume_ratio_20",
    "market_return_1",
    "market_return_5",
    "market_return_20",
    "market_mean_return_5",
    "market_volatility_20",
    "market_sma_gap_20",
    "relative_return_5",
    "relative_volatility_20",
    "weekday_sin",
    "weekday_cos",
]
CATEGORICAL_FEATURES = ["ticker"]
FEATURE_COLUMNS = CATEGORICAL_FEATURES + NUMERIC_FEATURES
CLASS_NAMES = {0: "down", 1: "flat", 2: "up"}


def classify_returns(
    values: pd.Series, threshold: float | pd.Series
) -> pd.Series:
    return pd.Series(
        np.where(values > threshold, 2, np.where(values < -threshold, 0, 1)),
        index=values.index,
        dtype="int64",
    )


def _market_frame(market_rows: list[dict] | None) -> pd.DataFrame:
    if not market_rows:
        return pd.DataFrame(
            columns=[
                "tradedate",
                "market_return_1",
                "market_return_5",
                "market_return_20",
                "market_mean_return_5",
                "market_volatility_20",
                "market_sma_gap_20",
            ]
        )
    market = pd.DataFrame(market_rows).sort_values("tradedate")
    market_close = market["close"].astype(float)
    market_return = market_close.pct_change()
    return pd.DataFrame(
        {
            "tradedate": market["tradedate"],
            "market_return_1": market_return,
            "market_return_5": market_close.pct_change(5),
            "market_return_20": market_close.pct_change(20),
            "market_mean_return_5": market_return.rolling(5).mean(),
            "market_volatility_20": market_return.rolling(20).std(),
            "market_sma_gap_20": (
                market_close / market_close.rolling(20).mean() - 1.0
            ),
        }
    )


def build_feature_frame(
    rows: list[dict],
    ticker: str,
    market_rows: list[dict] | None = None,
) -> pd.DataFrame:
    """Возвращает признаки для всех дат, используя данные не позднее даты строки."""
    if len(rows) < 30:
        return pd.DataFrame(columns=FEATURE_COLUMNS)

    frame = pd.DataFrame(rows).sort_values("tradedate").reset_index(drop=True)
    close = frame["close"].astype(float)
    open_price = frame["open"].fillna(frame["close"]).astype(float)
    high = frame["high"].fillna(frame["close"]).astype(float)
    low = frame["low"].fillna(frame["close"]).astype(float)
    volume = frame["volume"].fillna(0.0).astype(float)
    daily_return = close.pct_change()
    frame["_max_abs_return_20"] = (
        daily_return.abs().rolling(20).max()
    )

    frame["ticker"] = ticker
    for period in (1, 3, 5, 10, 20):
        frame[f"return_{period}"] = close.pct_change(period)
    for period in (5, 10):
        frame[f"mean_return_{period}"] = daily_return.rolling(period).mean()
    for period in (5, 10, 20):
        frame[f"volatility_{period}"] = daily_return.rolling(period).std()
        frame[f"sma_gap_{period}"] = close / close.rolling(period).mean() - 1.0

    delta = close.diff()
    average_gain = delta.clip(lower=0).rolling(14).mean()
    average_loss = (-delta.clip(upper=0)).rolling(14).mean()
    relative_strength = average_gain / average_loss.clip(lower=1e-12)
    frame["rsi_14"] = 100.0 - 100.0 / (1.0 + relative_strength)

    previous_close = close.shift(1)
    true_range = pd.concat(
        [
            high - low,
            (high - previous_close).abs(),
            (low - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    frame["atr_14"] = true_range.rolling(14).mean() / close
    frame["range_pct"] = (high - low) / close
    frame["overnight_gap"] = open_price / previous_close - 1.0
    frame["intraday_return"] = close / open_price - 1.0
    day_range = (high - low).replace(0.0, np.nan)
    frame["close_position"] = ((close - low) / day_range).fillna(0.5)

    log_volume = np.log1p(volume)
    volume_mean = log_volume.rolling(20).mean()
    volume_std = log_volume.rolling(20).std().replace(0.0, np.nan)
    frame["volume_change"] = volume.pct_change().replace(
        [np.inf, -np.inf], np.nan
    )
    frame["volume_z_20"] = (log_volume - volume_mean) / volume_std
    turnover = close * volume
    log_turnover = np.log1p(turnover)
    frame["log_turnover"] = log_turnover
    frame["turnover_change"] = turnover.pct_change().replace(
        [np.inf, -np.inf], np.nan
    )
    frame["turnover_z_20"] = (
        (log_turnover - log_turnover.rolling(20).mean())
        / log_turnover.rolling(20).std().replace(0.0, np.nan)
    )
    frame["zero_volume_ratio_20"] = (
        (volume <= 0).astype(float).rolling(20).mean()
    )

    frame = frame.merge(_market_frame(market_rows), on="tradedate", how="left")
    for column in (
        "market_return_1",
        "market_return_5",
        "market_return_20",
        "market_mean_return_5",
        "market_volatility_20",
        "market_sma_gap_20",
    ):
        frame[column] = frame[column].fillna(0.0)
    frame["relative_return_5"] = (
        frame["return_5"] - frame["market_return_5"]
    )
    frame["relative_volatility_20"] = (
        frame["volatility_20"]
        / frame["market_volatility_20"].replace(0.0, np.nan)
    ).fillna(1.0)

    dates = pd.to_datetime(frame["tradedate"])
    frame["weekday_sin"] = np.sin(
        2.0 * np.pi * dates.dt.day_of_week / 5.0
    )
    frame["weekday_cos"] = np.cos(
        2.0 * np.pi * dates.dt.day_of_week / 5.0
    )
    frame = frame.replace([np.inf, -np.inf], np.nan)
    return frame


def build_training_frame(
    rows: list[dict],
    ticker: str,
    market_rows: list[dict] | None,
    threshold: float,
    volatility_threshold_multiplier: float = 0.0,
    max_abs_return: float = 0.25,
) -> pd.DataFrame:
    """Добавляет доходность и класс следующего торгового дня."""
    frame = build_feature_frame(rows, ticker, market_rows)
    if frame.empty:
        return frame
    frame["current_price"] = frame["close"].astype(float)
    frame["target_price"] = frame["close"].shift(-1).astype(float)
    frame["target_open"] = frame["open"].shift(-1).astype(float)
    frame["target_date"] = frame["tradedate"].shift(-1)
    frame["target_return"] = frame["target_price"] / frame["current_price"] - 1.0
    frame["strategy_return"] = (
        frame["target_price"] / frame["target_open"] - 1.0
    )
    valid_target = frame["target_return"].notna()
    frame["target_threshold"] = np.maximum(
        threshold,
        frame["volatility_20"] * volatility_threshold_multiplier,
    )
    frame.loc[valid_target, "target_class"] = classify_returns(
        frame.loc[valid_target, "target_return"],
        frame.loc[valid_target, "target_threshold"],
    )
    required = FEATURE_COLUMNS + [
        "target_return",
        "strategy_return",
        "target_class",
        "target_date",
    ]
    valid = frame.dropna(subset=required)
    valid = valid[
        (valid["target_return"].abs() <= max_abs_return)
        & (valid["strategy_return"].abs() <= max_abs_return)
        & (valid["_max_abs_return_20"] <= max_abs_return)
    ]
    return valid.reset_index(drop=True)


def latest_feature_row(
    rows: list[dict],
    ticker: str,
    market_rows: list[dict] | None,
    max_abs_return: float = 0.25,
) -> pd.DataFrame:
    frame = build_feature_frame(rows, ticker, market_rows)
    valid = frame.dropna(subset=FEATURE_COLUMNS)
    valid = valid[valid["_max_abs_return_20"] <= max_abs_return]
    if valid.empty:
        raise ValueError("Недостаточно истории для расчёта признаков.")
    return valid.iloc[[-1]][FEATURE_COLUMNS]
