#!/usr/bin/env python3
"""
Backfill books.search_tsv for fast BM25 (GIN index on a tsvector column).

The expression index on to_tsvector(... array_to_string(...) ...) cannot be
created in Postgres (array_to_string is not IMMUTABLE). A stored tsvector
column avoids that and lets queries use idx_books_search_tsv.

Usage (from repo root; ``lib`` must be on ``PYTHONPATH``):

  PYTHONPATH=. python3 scripts/backfill_search_tsv.py
  PYTHONPATH=. python3 scripts/backfill_search_tsv.py --batch 10000 --max-batches 50
"""
from __future__ import annotations

import argparse

from lib.db import get_connection


def _row_tsvector_sql() -> str:
    """Same document shape as app/services/search.py bm25_search."""
    return """to_tsvector('english',
        coalesce(title, '') || ' ' ||
        coalesce(array_to_string(authors, ' '), '') || ' ' ||
        coalesce(synopsis, '') || ' ' ||
        coalesce(array_to_string(genres, ' '), '') || ' ' ||
        coalesce(array_to_string(subjects, ' '), '')
    )"""


def main() -> int:
    p = argparse.ArgumentParser(description="Backfill books.search_tsv in batches.")
    p.add_argument("--batch", type=int, default=5000, help="Rows per UPDATE batch")
    p.add_argument("--max-batches", type=int, default=0, help="Stop after N batches (0 = unlimited)")
    args = p.parse_args()

    expr = _row_tsvector_sql()
    sql = f"""
        UPDATE books AS b SET search_tsv = {expr}
        FROM (
            SELECT id FROM books
            WHERE search_tsv IS NULL
            LIMIT %(lim)s
        ) AS sub
        WHERE b.id = sub.id
    """

    conn = get_connection()
    total = 0
    try:
        batches = 0
        while True:
            if args.max_batches and batches >= args.max_batches:
                break
            with conn.cursor() as cur:
                cur.execute(sql, {"lim": args.batch})
                n = cur.rowcount
            conn.commit()
            total += n
            batches += 1
            print(f"batch {batches}: updated {n} rows (cumulative {total})")
            if n == 0:
                break
    finally:
        conn.close()

    print(f"Done. Total rows updated: {total}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
