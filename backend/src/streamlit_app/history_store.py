import sqlite3
from datetime import datetime, timezone
from typing import Any


def init_db(db_path: str) -> None:
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS search_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                query TEXT NOT NULL,
                domain TEXT NOT NULL,
                method TEXT NOT NULL,
                top_k INTEGER NOT NULL,
                result_count INTEGER NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            DELETE FROM search_history
            WHERE id NOT IN (
                SELECT MAX(id)
                FROM search_history
                GROUP BY user_id, lower(query)
            )
            """
        )
        conn.execute("DROP INDEX IF EXISTS idx_search_history_user_query")
        conn.execute(
            """
            CREATE UNIQUE INDEX IF NOT EXISTS idx_search_history_user_query_nocase
            ON search_history(user_id, query COLLATE NOCASE)
            """
        )
        conn.commit()


def save_search(
    db_path: str,
    user_id: str,
    query: str,
    domain: str,
    method: str,
    top_k: int,
    result_count: int,
) -> None:
    normalized_query = " ".join(str(query).split())
    if not normalized_query:
        return

    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO search_history (user_id, query, domain, method, top_k, result_count, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(user_id, query) DO UPDATE SET
                domain = excluded.domain,
                method = excluded.method,
                top_k = excluded.top_k,
                result_count = excluded.result_count,
                created_at = excluded.created_at
            """,
            (
                user_id,
                normalized_query,
                domain,
                method,
                int(top_k),
                int(result_count),
                datetime.now(timezone.utc).isoformat(),
            ),
        )
        conn.execute(
            """
            DELETE FROM search_history
            WHERE id NOT IN (
                SELECT MAX(id)
                FROM search_history
                GROUP BY user_id, lower(query)
            )
            """
        )
        conn.commit()


def list_recent_searches(db_path: str, user_id: str, limit: int = 20) -> list[dict[str, Any]]:
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """
            SELECT id, query, domain, method, top_k, result_count, created_at
            FROM search_history
            WHERE user_id = ?
            ORDER BY created_at DESC, id DESC
            LIMIT ?
            """,
            (user_id, int(limit)),
        ).fetchall()

    return [dict(row) for row in rows]


def clear_history(db_path: str, user_id: str) -> None:
    with sqlite3.connect(db_path) as conn:
        conn.execute("DELETE FROM search_history WHERE user_id = ?", (user_id,))
        conn.commit()


def list_all_searches(db_path: str, limit: int = 500) -> list[dict[str, Any]]:
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """
            SELECT id, user_id, query, domain, method, top_k, result_count, created_at
            FROM search_history
            ORDER BY created_at DESC, id DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
    return [dict(row) for row in rows]
