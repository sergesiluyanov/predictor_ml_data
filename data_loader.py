"""Параллельная загрузка и локальный кэш истории MOEX."""

from concurrent.futures import as_completed, ThreadPoolExecutor
import json
from pathlib import Path
import threading
import time

from moex_client import (
    fetch_index_last_n_days,
    fetch_last_n_days,
)


_PRINT_LOCK = threading.Lock()


def _cache_is_fresh(path: Path, max_age_hours: int = 20) -> bool:
    if not path.is_file():
        return False
    age_seconds = time.time() - path.stat().st_mtime
    return age_seconds < max_age_hours * 3600


def _read_cache(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_cache(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(rows, ensure_ascii=False),
        encoding="utf-8",
    )
    temporary.replace(path)


def load_index_history(
    days: int,
    cache_dir: str | Path = "data/cache",
    refresh: bool = False,
) -> list[dict]:
    path = Path(cache_dir) / f"IMOEX_{days}.json"
    if not refresh and _cache_is_fresh(path):
        return _read_cache(path)
    rows = fetch_index_last_n_days("IMOEX", days)
    if rows:
        _write_cache(path, rows)
    return rows


def _load_one(
    ticker: str,
    days: int,
    cache_dir: Path,
    refresh: bool,
) -> tuple[str, list[dict]]:
    path = cache_dir / f"{ticker}_{days}.json"
    if not refresh and _cache_is_fresh(path):
        return ticker, _read_cache(path)
    rows = fetch_last_n_days(ticker, days)
    if rows:
        _write_cache(path, rows)
    return ticker, rows


def load_ticker_histories(
    tickers: list[str],
    days: int,
    workers: int = 8,
    cache_dir: str | Path = "data/cache",
    refresh: bool = False,
) -> tuple[dict[str, list[dict]], list[dict]]:
    """Загружает тикеры параллельно; повторный запуск использует свежий кэш."""
    cache_path = Path(cache_dir)
    histories: dict[str, list[dict]] = {}
    errors = []
    completed = 0

    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        futures = {
            executor.submit(
                _load_one,
                ticker,
                days,
                cache_path,
                refresh,
            ): ticker
            for ticker in tickers
        }
        for future in as_completed(futures):
            ticker = futures[future]
            try:
                _, rows = future.result()
                histories[ticker] = rows
            except Exception as exc:
                errors.append({"ticker": ticker, "reason": str(exc)})
            completed += 1
            if completed % 10 == 0 or completed == len(tickers):
                with _PRINT_LOCK:
                    print(
                        f"Загружено {completed}/{len(tickers)} тикеров",
                        flush=True,
                    )
    return histories, errors
