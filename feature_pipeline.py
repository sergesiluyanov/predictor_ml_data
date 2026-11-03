"""Единая подготовка признаков для обучения, бэктеста и инференса."""

from dataclasses import dataclass
from datetime import datetime

import numpy as np


SEQ_LEN = 30
FEATURES = 4
OUTPUT_DIM = 1


@dataclass(frozen=True)
class SupervisedDataset:
    X: np.ndarray
    y: np.ndarray
    target_dates: list[str]
    current_prices: np.ndarray
    target_prices: np.ndarray


def _arrays(rows: list[dict]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    closes = np.asarray([row["close"] for row in rows], dtype=np.float32)
    volumes = np.asarray(
        [row.get("volume") or 0.0 for row in rows], dtype=np.float32
    )
    returns = np.zeros_like(closes)
    returns[1:] = (closes[1:] - closes[:-1]) / (closes[:-1] + 1e-10)
    return closes, volumes, returns


def _feature_window(
    rows: list[dict],
    closes: np.ndarray,
    volumes: np.ndarray,
    returns: np.ndarray,
    current_index: int,
    seq_len: int,
) -> np.ndarray:
    start = current_index - seq_len + 1
    close_window = closes[start : current_index + 1]
    volume_window = volumes[start : current_index + 1]
    volume_max = float(volume_window.max()) or 1.0
    weekdays = np.asarray(
        [
            datetime.strptime(rows[index]["tradedate"], "%Y-%m-%d").weekday()
            / 7.0
            for index in range(start, current_index + 1)
        ],
        dtype=np.float32,
    )
    return np.stack(
        [
            close_window / close_window[-1],
            returns[start : current_index + 1],
            volume_window / volume_max,
            weekdays,
        ],
        axis=-1,
    ).astype(np.float32)


def build_supervised_dataset(
    rows: list[dict], seq_len: int = SEQ_LEN
) -> SupervisedDataset | None:
    """Строит выборку: известные данные до дня i → доходность дня i+1."""
    if len(rows) < seq_len + 2:
        return None

    closes, volumes, returns = _arrays(rows)
    features, targets = [], []
    target_dates, current_prices, target_prices = [], [], []

    # Начинаем с seq_len, чтобы первая доходность внутри окна была реальной,
    # а не искусственным нулём в returns[0].
    for current_index in range(seq_len, len(rows) - 1):
        features.append(
            _feature_window(
                rows,
                closes,
                volumes,
                returns,
                current_index,
                seq_len,
            )
        )
        target_index = current_index + 1
        targets.append(returns[target_index])
        target_dates.append(rows[target_index]["tradedate"])
        current_prices.append(closes[current_index])
        target_prices.append(closes[target_index])

    if not features:
        return None
    return SupervisedDataset(
        X=np.stack(features),
        y=np.asarray(targets, dtype=np.float32),
        target_dates=target_dates,
        current_prices=np.asarray(current_prices, dtype=np.float32),
        target_prices=np.asarray(target_prices, dtype=np.float32),
    )


def build_latest_input(
    rows: list[dict], seq_len: int = SEQ_LEN
) -> np.ndarray:
    """Готовит окно для прогноза следующего ещё неизвестного дня."""
    if len(rows) < seq_len + 1:
        raise ValueError(
            f"Недостаточно истории: нужно минимум {seq_len + 1} торговых дней."
        )
    closes, volumes, returns = _arrays(rows)
    features = _feature_window(
        rows,
        closes,
        volumes,
        returns,
        len(rows) - 1,
        seq_len,
    )
    return features.reshape(1, seq_len, FEATURES)
