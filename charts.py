"""Построение PNG-графиков для Telegram."""

from io import BytesIO
from datetime import datetime

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt

from predictor import StockPrediction


def build_prediction_chart(
    prediction: StockPrediction, points: int = 90
) -> bytes:
    history = prediction.history[-points:]
    dates = [
        datetime.strptime(row["tradedate"], "%Y-%m-%d") for row in history
    ]
    prices = [row["close"] for row in history]
    forecast_date = datetime.strptime(prediction.forecast_date, "%Y-%m-%d")

    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.plot(dates, prices, color="#2878b5", linewidth=2, label="Цена закрытия")
    ax.plot(
        [dates[-1], forecast_date],
        [prices[-1], prediction.predicted_price],
        color="#e45756",
        linestyle="--",
        linewidth=2,
        marker="o",
        label="Прогноз",
    )
    ax.set_title(f"{prediction.ticker}: история и прогноз")
    ax.set_ylabel("Цена, ₽")
    ax.grid(alpha=0.25)
    ax.legend()
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d.%m.%Y"))
    fig.autofmt_xdate()
    fig.tight_layout()

    output = BytesIO()
    fig.savefig(output, format="png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    return output.getvalue()
