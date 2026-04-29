from __future__ import annotations

from psycopg2.extras import RealDictCursor
from lib.db import get_connection


def execute_query(sql: str, params: dict | tuple | list | None = None) -> list[dict]:
    """Run a read query, return rows as dicts."""
    conn = get_connection()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(sql, params)
            return [dict(row) for row in cur.fetchall()]
    finally:
        conn.close()


def execute_write(sql: str, params: dict | tuple | list | None = None) -> int:
    """Run a write query (INSERT/UPDATE), return rowcount."""
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(sql, params)
            conn.commit()
            return cur.rowcount
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
