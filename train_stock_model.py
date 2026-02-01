"""
Обучение модели прогноза цен акций на данных MOEX и экспорт в TFLite.
Использует последовательность последних N дней (close, returns, объём) → следующий день (return).
"""

import argparse
import os
import sys
from datetime import datetime

try:
    import numpy as np
except ImportError:
    sys.exit(
        "Не найден numpy. Запускайте скрипт из активированного venv:\n"
        "  source .venv/bin/activate\n"
        "  python train_stock_model.py ...\n"
        "или явно: .venv/bin/python train_stock_model.py ..."
    )

try:
    import tensorflow as tf
except ImportError:
    raise SystemExit("Установите TensorFlow: pip install tensorflow")

from moex_client import (
    fetch_stock_history,
    fetch_last_n_days,
    fetch_liquid_tickers_full,
    DEFAULT_LIQUID_TICKERS,
)


# Параметры модели (должны совпадать с приложением при использовании)
SEQ_LEN = 30          # сколько дней в окне
FEATURES = 4          # close_norm, return, volume_norm, weekday
OUTPUT_DIM = 1        # предсказание: дневная доходность на следующий день


def build_dataset_fixed(rows: list[dict], seq_len: int = SEQ_LEN):
    """Строит X и y; weekday берётся из tradedate."""
    if len(rows) < seq_len + 2:
        return None, None

    closes = np.array([r["close"] for r in rows], dtype=np.float32)
    volumes = np.array([r.get("volume") or 0.0 for r in rows], dtype=np.float32)

    returns = np.zeros_like(closes)
    returns[1:] = (closes[1:] - closes[:-1]) / (closes[:-1] + 1e-10)

    vol_max = volumes.max() or 1.0

    X_list, y_list = [], []
    for i in range(seq_len, len(rows) - 1):
        window_close = closes[i - seq_len : i + 1]
        close_norm = (window_close / window_close[-1]).astype(np.float32)
        ret_window = returns[i - seq_len : i]
        vol_window = (volumes[i - seq_len : i] / vol_max).astype(np.float32)
        weekdays = np.array(
            [datetime.strptime(rows[j]["tradedate"], "%Y-%m-%d").weekday() / 7.0 for j in range(i - seq_len, i)],
            dtype=np.float32,
        )
        feats = np.stack([close_norm[:-1], ret_window, vol_window, weekdays], axis=-1)
        X_list.append(feats)
        y_list.append(returns[i + 1])

    if not X_list:
        return None, None
    return np.stack(X_list), np.array(y_list, dtype=np.float32)


def create_model(seq_len: int = SEQ_LEN, features: int = FEATURES):
    """Простая модель: LSTM + Dense → один выход (return)."""
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(seq_len, features)),
        tf.keras.layers.LSTM(32, return_sequences=False),
        tf.keras.layers.Dense(16, activation="relu"),
        tf.keras.layers.Dense(OUTPUT_DIM),
    ])
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    return model


def main():
    parser = argparse.ArgumentParser(description="Train stock prediction model and export TFLite")
    parser.add_argument("--ticker", default=None, help="Один тикер MOEX (если не задан --tickers)")
    parser.add_argument(
        "--tickers",
        default=None,
        help="Тикеры: 'all' (все с MOEX) или через запятую: SBER,GAZP,LKOH",
    )
    parser.add_argument(
        "--max-tickers",
        type=int,
        default=500,
        help="Макс. число тикеров при --tickers all (по умолчанию 500)",
    )
    parser.add_argument("--days", type=int, default=730, help="Сколько дней истории загружать")
    parser.add_argument("--epochs", type=int, default=20, help="Эпохи обучения")
    parser.add_argument("--out", default="stock_model.tflite", help="Путь к выходному .tflite")
    parser.add_argument("--from", dest="from_date", default=None, help="Начало периода YYYY-MM-DD")
    parser.add_argument("--till", dest="till_date", default=None, help="Конец периода YYYY-MM-DD")
    args = parser.parse_args()

    # Список тикеров для обучения
    if args.tickers is not None:
        if args.tickers.strip().lower() == "all":
            print("Загрузка списка ликвидных тикеров (площадка TQBR) с MOEX...")
            try:
                tickers = fetch_liquid_tickers_full()
            except Exception as e:
                print(f"API списка тикеров недоступен ({e}), используем встроенный список.")
                tickers = []
            if not tickers:
                tickers = list(DEFAULT_LIQUID_TICKERS)[: args.max_tickers]
                print(f"Используем встроенный список ликвидных тикеров: {len(tickers)} шт.")
            else:
                tickers = tickers[: args.max_tickers]
                print(f"Ликвидных тикеров (TQBR) для загрузки: {len(tickers)}")
        else:
            tickers = [t.strip() for t in args.tickers.split(",") if t.strip()]
    else:
        tickers = [args.ticker or "SBER"]

    min_rows = SEQ_LEN + 10
    X_parts, y_parts = [], []
    failed, skipped = [], []

    for i, ticker in enumerate(tickers):
        if (i + 1) % 50 == 0 or i == 0:
            print(f"Загрузка данных: {i + 1}/{len(tickers)} — {ticker}...")
        try:
            if args.from_date and args.till_date:
                rows = fetch_stock_history(ticker, args.from_date, args.till_date)
            else:
                rows = fetch_last_n_days(ticker, args.days)
        except Exception as e:
            failed.append((ticker, str(e)))
            continue
        if len(rows) < min_rows:
            skipped.append((ticker, len(rows)))
            continue
        X_t, y_t = build_dataset_fixed(rows, SEQ_LEN)
        if X_t is not None and y_t is not None:
            X_parts.append(X_t)
            y_parts.append(y_t)

    if not X_parts:
        raise SystemExit(
            "Нет данных ни по одному тикеру. "
            f"Пропущено (мало данных): {len(skipped)}, ошибки: {len(failed)}. "
            "Проверьте --days или --tickers."
        )

    X = np.concatenate(X_parts, axis=0)
    y = np.concatenate(y_parts, axis=0)
    # Перемешиваем для обучения (по срезам от разных тикеров)
    rng = np.random.default_rng(42)
    idx = rng.permutation(len(y))
    X, y = X[idx], y[idx]

    if failed:
        print(f"Ошибки загрузки ({len(failed)}): {failed[:5]}{'...' if len(failed) > 5 else ''}")
    if skipped:
        print(f"Пропущено из-за малого числа записей: {len(skipped)} тикеров")
    print(f"Dataset: X {X.shape}, y {y.shape} (тикеров использовано: {len(X_parts)})")

    model = create_model(SEQ_LEN, FEATURES)
    model.fit(X, y, epochs=args.epochs, validation_split=0.2, verbose=1)

    # Экспорт в TFLite (LSTM требует Select TF ops для TensorList)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,
        tf.lite.OpsSet.SELECT_TF_OPS,
    ]
    converter._experimental_lower_tensor_list_ops = False
    tflite_model = converter.convert()
    out_path = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "wb") as f:
        f.write(tflite_model)
    print(f"Saved: {out_path}")
    print("Скопируйте .tflite в Flutter: assets/models/stock_model.tflite")


if __name__ == "__main__":
    main()
