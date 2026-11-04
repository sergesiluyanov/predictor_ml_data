"""Постоянный лимит бесплатных запросов Telegram-пользователей."""

import sqlite3
from pathlib import Path


def _connect(database_path: str | Path) -> sqlite3.Connection:
    path = Path(database_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path, timeout=30)
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS user_usage (
            user_id INTEGER PRIMARY KEY,
            request_count INTEGER NOT NULL DEFAULT 0,
            updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    return connection


def consume_request(
    database_path: str | Path,
    user_id: int,
    limit: int,
) -> tuple[bool, int]:
    """Атомарно резервирует запрос и возвращает оставшийся баланс."""
    with _connect(database_path) as connection:
        connection.execute("BEGIN IMMEDIATE")
        row = connection.execute(
            "SELECT request_count FROM user_usage WHERE user_id = ?",
            (user_id,),
        ).fetchone()
        used = int(row[0]) if row else 0
        if used >= limit:
            connection.rollback()
            return False, 0
        new_count = used + 1
        connection.execute(
            """
            INSERT INTO user_usage (user_id, request_count, updated_at)
            VALUES (?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(user_id) DO UPDATE SET
                request_count = excluded.request_count,
                updated_at = CURRENT_TIMESTAMP
            """,
            (user_id, new_count),
        )
        connection.commit()
        return True, limit - new_count


def refund_request(database_path: str | Path, user_id: int) -> None:
    """Возвращает запрос при технической ошибке или неизвестном тикере."""
    with _connect(database_path) as connection:
        connection.execute(
            """
            UPDATE user_usage
            SET request_count = MAX(0, request_count - 1),
                updated_at = CURRENT_TIMESTAMP
            WHERE user_id = ?
            """,
            (user_id,),
        )
