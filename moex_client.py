"""
MOEX ISS API client — получение исторических данных с Московской биржи.
Соответствует логике Flutter-приложения (stock_service.dart).
"""

import time
import requests
from datetime import datetime, timedelta
from typing import Optional

# Повторы при сбоях сети (Connection reset, timeout)
REQUEST_RETRIES = 3
REQUEST_RETRY_DELAY = 2


BASE_URL = "https://iss.moex.com/iss/history/engines/stock/markets/shares/securities"
SECURITIES_LIST_URL = "https://iss.moex.com/iss/engines/stock/markets/shares/securities.json"
PAGE_SIZE = 100
LIST_PAGE_SIZE = 100


def parse_date(date_str: str) -> datetime:
    """Парсинг даты из строки (YYYY-MM-DD, DD.MM.YYYY, YYYYMMDD)."""
    date_str = date_str.strip()
    try:
        if "-" in date_str and len(date_str) == 10:
            return datetime.strptime(date_str, "%Y-%m-%d")
        if "." in date_str:
            return datetime.strptime(date_str, "%d.%m.%Y")
        if "/" in date_str:
            return datetime.strptime(date_str, "%d/%m/%Y")
        if len(date_str) == 8 and date_str.isdigit():
            return datetime.strptime(date_str, "%Y%m%d")
    except ValueError:
        pass
    return datetime.now()


def fetch_stock_history(
    ticker: str,
    from_date: str,
    till_date: str,
) -> list[dict]:
    """
    Загружает историю торгов по тикеру с MOEX.
    Возвращает список записей с полями: tradedate, close, volume, high, low.
    """
    start = parse_date(from_date)
    end = parse_date(till_date)
    yesterday = datetime.now() - timedelta(days=1)
    if start > yesterday or end > yesterday:
        return []

    all_rows = []
    page_start = 0

    while True:
        url = (
            f"{BASE_URL}/{ticker}.json"
            f"?from={from_date}&till={till_date}&start={page_start}"
            f"&iss.meta=off&iss.json=extended"
        )
        resp = requests.get(url, timeout=30)
        if resp.status_code != 200:
            break

        data = resp.json()
        if not isinstance(data, list) or len(data) < 2:
            break

        history_block = data[1]
        if not isinstance(history_block, dict) or "history" not in history_block:
            break

        history = history_block["history"]
        if not history:
            break

        for item in history:
            tradedate = item.get("TRADEDATE")
            close = item.get("CLOSE")
            if not tradedate or close is None or float(close) <= 0:
                continue
            try:
                all_rows.append({
                    "tradedate": str(tradedate),
                    "close": float(close),
                    "volume": float(item["VOLUME"]) if item.get("VOLUME") is not None else None,
                    "high": float(item["HIGH"]) if item.get("HIGH") is not None else None,
                    "low": float(item["LOW"]) if item.get("LOW") is not None else None,
                })
            except (TypeError, ValueError):
                continue

        if len(history) < PAGE_SIZE:
            break
        page_start += PAGE_SIZE

    # Дедупликация по дате (оставляем запись с большим объёмом — основная сессия)
    by_date = {}
    for row in all_rows:
        d = row["tradedate"]
        if d not in by_date or (row.get("volume") or 0) > (by_date[d].get("volume") or 0):
            by_date[d] = row

    result = [by_date[d] for d in sorted(by_date)]
    return result


def _get_with_retries(url: str, timeout: int = 30) -> requests.Response | None:
    """GET с повторами при ConnectionError / timeout."""
    for attempt in range(REQUEST_RETRIES):
        try:
            resp = requests.get(url, timeout=timeout)
            return resp
        except (requests.ConnectionError, requests.exceptions.ConnectTimeout, OSError) as e:
            if attempt < REQUEST_RETRIES - 1:
                time.sleep(REQUEST_RETRY_DELAY * (attempt + 1))
            else:
                raise e
    return None


def fetch_all_tickers(start: int = 0) -> list[str]:
    """
    Загружает список тикеров (SECID) с рынка акций MOEX.
    Пагинация: start=0, 100, 200, ...
    """
    url = f"{SECURITIES_LIST_URL}?start={start}"
    resp = _get_with_retries(url)
    if resp is None or resp.status_code != 200:
        return []
    data = resp.json()
    if not isinstance(data, dict) or "securities" not in data:
        return []
    block = data["securities"]
    if not isinstance(block, dict) or "columns" not in block or "data" not in block:
        return []
    columns = block["columns"]
    rows = block["data"]
    try:
        idx = columns.index("SECID")
    except ValueError:
        return []
    return [str(row[idx]) for row in rows if len(row) > idx and row[idx]]


def fetch_all_tickers_full() -> list[str]:
    """Загружает полный список тикеров с MOEX (все страницы). При ошибке сети — пустой список."""
    all_tickers = []
    start = 0
    try:
        while True:
            batch = fetch_all_tickers(start)
            if not batch:
                break
            all_tickers.extend(batch)
            if len(batch) < LIST_PAGE_SIZE:
                break
            start += LIST_PAGE_SIZE
    except Exception:
        pass
    return all_tickers


# Запасной список ликвидных тикеров MOEX, если API списка недоступен
DEFAULT_LIQUID_TICKERS = [
    "SBER", "GAZP", "LKOH", "GMKN", "NVTK", "YNDX", "ROSN", "TATN", "PLZL",
    "MAGN", "ALRS", "NLMK", "MTSS", "MOEX", "SNGS", "VTBR", "TCSG", "POLY",
    "AFKS", "AFLT", "CBOM", "CHMF", "IRAO", "MGNT", "MTLR", "RUAL", "SBERP",
    "SIBN", "TATNP", "TRNFP", "DSKY", "FIVE", "OZON", "PIKK", "POSI",
    "RENI", "SMLT", "UPRO", "FIXP", "LSRG", "PHOR", "RTKM", "SGZH", "ENRU",
]


def fetch_last_n_days(ticker: str, days: int = 365) -> list[dict]:
    """Загружает последние N календарных дней истории (удобно для обучения)."""
    end = datetime.now() - timedelta(days=1)
    start = end - timedelta(days=days)
    return fetch_stock_history(
        ticker,
        start.strftime("%Y-%m-%d"),
        end.strftime("%Y-%m-%d"),
    )


if __name__ == "__main__":
    # Пример: загрузка данных по Сберу
    rows = fetch_stock_history("SBER", "2024-01-01", "2024-12-31")
    print(f"Loaded {len(rows)} records")
    if rows:
        print("Sample:", rows[-3:])

