from __future__ import annotations

from psycopg2.extras import RealDictCursor
from lib.db import get_connection


def execute_query(sql: str, params: dict | tuple | list | None = None) -> list[dict]:
    """Run a SELECT, return rows as dicts."""
    conn = get_connection()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            # Hard cap per query - prevents stray BM25 full-table scans from
            # hanging the request when search_tsv isn't backfilled yet.
            cur.execute("SET statement_timeout = '10s'")
            cur.execute(sql, params)
            return [dict(row) for row in cur.fetchall()]
    finally:
        conn.close()


def execute_write(sql: str, params: dict | tuple | list | None = None) -> int:
    """Run an INSERT/UPDATE/DELETE, return rowcount."""
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


def execute_write_fetch(sql: str, params: dict | tuple | list | None = None) -> list[dict]:
    """INSERT/UPDATE/DELETE with RETURNING clause."""
    conn = get_connection()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(sql, params)
            rows = [dict(row) for row in cur.fetchall()]
        conn.commit()
        return rows
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
