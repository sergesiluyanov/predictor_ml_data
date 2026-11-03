"""Telegram-бот с прогнозом акций MOEX."""

import asyncio
from html import escape
import json
import logging
import os
from pathlib import Path

from aiogram import Bot, Dispatcher
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode
from aiogram.filters import Command, CommandStart
from aiogram.types import BufferedInputFile, Message
from dotenv import load_dotenv

from charts import build_prediction_chart
from predictor import (
    PredictionError,
    StockPrediction,
    normalize_ticker,
    predict_stock,
)


load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)
dispatcher = Dispatcher()


def _command_argument(message: Message) -> str | None:
    parts = (message.text or "").split(maxsplit=1)
    return parts[1] if len(parts) == 2 else None


def _format_price(value: float) -> str:
    return f"{value:,.2f}".replace(",", " ").replace(".", ",")


def _forecast_text(prediction: StockPrediction) -> str:
    change = prediction.predicted_return * 100
    arrow = "📈" if change >= 0 else "📉"
    probability_text = ""
    if prediction.probabilities:
        confidence = max(prediction.probabilities.values())
        signal_threshold = prediction.signal_threshold or 0.55
        signal_names = {
            "up": "рост",
            "flat": "боковик",
            "down": "падение",
        }
        if confidence >= signal_threshold and prediction.signal:
            signal_text = (
                f"Сигнал: <b>{signal_names[prediction.signal]}</b> "
                f"({confidence:.1%})"
            )
        else:
            signal_text = "Уверенный сигнал отсутствует"
        probability_text = (
            "\nВероятности: "
            f"рост {prediction.probabilities['up']:.1%}, "
            f"боковик {prediction.probabilities['flat']:.1%}, "
            f"падение {prediction.probabilities['down']:.1%}\n"
            f"{signal_text}\n"
        )
    trading_text = ""
    if prediction.trading_signal:
        trading_names = {
            "up": "рост",
            "down": "падение",
            "none": "нет подтверждённого сигнала",
        }
        confidence_text = (
            f" ({prediction.trading_confidence:.1%})"
            if prediction.trading_signal != "none"
            and prediction.trading_confidence is not None
            else ""
        )
        trading_text = (
            "\nСигнал на следующую сессию: "
            f"<b>{trading_names[prediction.trading_signal]}</b>"
            f"{confidence_text}\n"
        )
    return (
        f"<b>{prediction.ticker}</b>\n"
        f"Цена на {prediction.last_date}: "
        f"<b>{_format_price(prediction.last_price)} ₽</b>\n"
        f"Прогноз на {prediction.forecast_date}: "
        f"<b>{_format_price(prediction.predicted_price)} ₽</b>\n"
        f"{arrow} Ожидаемое изменение: <b>{change:+.2f}%</b>"
        f"{probability_text}{trading_text}\n"
        "⚠️ Прогноз модели не является инвестиционной рекомендацией."
    )


async def _get_prediction(message: Message, ticker: str) -> StockPrediction | None:
    try:
        normalized_ticker = normalize_ticker(ticker)
    except PredictionError as exc:
        await message.answer(f"Не удалось построить прогноз:\n{escape(str(exc))}")
        return None

    status = await message.answer(
        f"Загружаю данные MOEX для {normalized_ticker}…"
    )
    try:
        prediction = await asyncio.to_thread(predict_stock, normalized_ticker)
    except PredictionError as exc:
        await status.edit_text(
            f"Не удалось построить прогноз:\n{escape(str(exc))}"
        )
        return None
    except Exception:
        logger.exception("Unexpected prediction error")
        await status.edit_text("Внутренняя ошибка при построении прогноза.")
        return None
    await status.delete()
    return prediction


@dispatcher.message(CommandStart())
async def start_handler(message: Message) -> None:
    await message.answer(
        "Отправьте тикер акции MOEX или используйте команды:\n"
        "<code>/predict SBER</code> — прогноз цены\n"
        "<code>/chart SBER</code> — прогноз с графиком\n"
        "<code>/ideas</code> — проверенные long-кандидаты на 5 сессий\n"
        "<code>/help</code> — справка"
    )


@dispatcher.message(Command("help"))
async def help_handler(message: Message) -> None:
    await start_handler(message)


@dispatcher.message(Command("ideas"))
async def ideas_handler(message: Message) -> None:
    path = Path("models/latest_long_ideas.json")
    if not path.is_file():
        await message.answer(
            "Список long-кандидатов ещё не сформирован. "
            "Запустите <code>generate_long_ideas.py</code>."
        )
        return
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        logger.exception("Cannot read long ideas")
        await message.answer("Не удалось прочитать список long-кандидатов.")
        return
    candidates = payload.get("candidates", [])
    if candidates:
        lines = [
            f"{item['rank']}. <b>{escape(item['ticker'])}</b>"
            for item in candidates[:10]
        ]
        candidates_text = "\n".join(lines)
    else:
        candidates_text = "Подходящих кандидатов сейчас нет — режим cash."
    await message.answer(
        f"<b>Long-кандидаты на 5 сессий</b>\n"
        f"Данные на {escape(str(payload.get('as_of', '—')))}\n"
        f"{candidates_text}\n\n"
        "Сигнал прошёл исторический walk-forward gate, "
        "но ещё проходит live-проверку.\n"
        "⚠️ Не является инвестиционной рекомендацией."
    )


@dispatcher.message(Command("predict"))
async def predict_handler(message: Message) -> None:
    ticker = _command_argument(message)
    if not ticker:
        await message.answer("Укажите тикер: <code>/predict SBER</code>")
        return
    prediction = await _get_prediction(message, ticker)
    if prediction:
        await message.answer(_forecast_text(prediction))


@dispatcher.message(Command("chart"))
async def chart_handler(message: Message) -> None:
    ticker = _command_argument(message)
    if not ticker:
        await message.answer("Укажите тикер: <code>/chart SBER</code>")
        return
    prediction = await _get_prediction(message, ticker)
    if not prediction:
        return
    image = await asyncio.to_thread(build_prediction_chart, prediction)
    await message.answer_photo(
        BufferedInputFile(image, filename=f"{prediction.ticker}_forecast.png"),
        caption=_forecast_text(prediction),
    )


@dispatcher.message()
async def ticker_handler(message: Message) -> None:
    ticker = (message.text or "").strip()
    if not ticker or ticker.startswith("/"):
        return
    prediction = await _get_prediction(message, ticker)
    if prediction:
        await message.answer(_forecast_text(prediction))


async def main() -> None:
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    if not token:
        raise SystemExit(
            "Не задан TELEGRAM_BOT_TOKEN. Скопируйте .env.example в .env "
            "и добавьте токен BotFather."
        )
    bot = Bot(
        token=token,
        default=DefaultBotProperties(parse_mode=ParseMode.HTML),
    )
    await dispatcher.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())
