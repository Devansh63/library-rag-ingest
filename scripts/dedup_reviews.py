"""Remove duplicate rows from the reviews table, keeping the lowest id per group.

Duplicates were introduced by a --reviews-only re-stream pass that did not
correctly honour the per-book cap for isbn-path reviews. The result is ~3.3M
extra rows where (isbn13, book_id, user_id, date_posted, review_text) repeats.

Dedup key: (isbn13, book_id, user_id, date_posted, review_text)
Keep:       lowest id in each group (first inserted)
Delete:     all higher ids in the same group

The deletion runs in batches of BATCH_SIZE to avoid a single massive
transaction that could time out or lock the table for minutes on Neon.

Usage:
    uv run python scripts/dedup_reviews.py --dry-run   # show counts, no writes
    uv run python scripts/dedup_reviews.py             # delete duplicates
    uv run python scripts/dedup_reviews.py --batch-size 5000
"""

from __future__ import annotations

import argparse
import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from lib.db import get_connection

BATCH_SIZE = 2000


def count_duplicates(conn) -> tuple[int, int]:
    """Return (duplicate_groups, duplicate_rows) - rows that will be deleted."""
    with conn.cursor() as cur:
        cur.execute("""
            SELECT
                COUNT(*) AS dup_groups,
                COALESCE(SUM(cnt - 1), 0) AS dup_rows
            FROM (
                SELECT COUNT(*) AS cnt
                FROM reviews
                GROUP BY isbn13, book_id, user_id, date_posted, review_text
                HAVING COUNT(*) > 1
            ) sub
        """)
        row = cur.fetchone()
    return int(row[0]), int(row[1])


def dedup_reviews(conn, batch_size: int, dry_run: bool) -> int:
    """Delete duplicate review rows in batches, keeping the lowest id per group."""
    print("Building list of duplicate ids to delete...")
    with conn.cursor() as cur:
        cur.execute("""
            CREATE TEMP TABLE reviews_to_delete AS
            SELECT id
            FROM reviews
            WHERE id NOT IN (
                SELECT MIN(id)
                FROM reviews
                GROUP BY isbn13, book_id, user_id, date_posted, review_text
            )
        """)
        cur.execute("SELECT COUNT(*) FROM reviews_to_delete")
        total_to_delete = cur.fetchone()[0]
    conn.commit()

    print(f"Found {total_to_delete:,} rows to delete.")

    if dry_run or total_to_delete == 0:
        with conn.cursor() as cur:
            cur.execute("DROP TABLE IF EXISTS reviews_to_delete")
        conn.commit()
        return total_to_delete

    deleted = 0
    batch_num = 0
    while True:
        with conn.cursor() as cur:
            cur.execute("SELECT id FROM reviews_to_delete LIMIT %s", (batch_size,))
            batch_ids = [row[0] for row in cur.fetchall()]

        if not batch_ids:
            break

        with conn.cursor() as cur:
            cur.execute("DELETE FROM reviews WHERE id = ANY(%s)", (batch_ids,))
            rows_deleted = cur.rowcount
            cur.execute("DELETE FROM reviews_to_delete WHERE id = ANY(%s)", (batch_ids,))

        conn.commit()
        deleted += rows_deleted
        batch_num += 1
        remaining = total_to_delete - deleted
        print(f"  Batch {batch_num}: deleted {rows_deleted:,} rows "
              f"({deleted:,} total, {remaining:,} remaining)")

    with conn.cursor() as cur:
        cur.execute("DROP TABLE IF EXISTS reviews_to_delete")
    conn.commit()
    return deleted


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dry-run", action="store_true",
                        help="Count duplicates without deleting anything.")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help=f"Rows to delete per transaction (default {BATCH_SIZE}).")
    args = parser.parse_args()

    conn = get_connection()

    print("Counting duplicates...")
    dup_groups, dup_rows = count_duplicates(conn)
    print(f"  Duplicate groups: {dup_groups:,}")
    print(f"  Rows to delete:   {dup_rows:,}")

    if dup_rows == 0:
        print("No duplicates found. Nothing to do.")
        conn.close()
        return 0

    if args.dry_run:
        print("Dry run - no deletions performed.")
        conn.close()
        return 0

    print(f"\nDeleting in batches of {args.batch_size:,}...")
    deleted = dedup_reviews(conn, args.batch_size, dry_run=False)
    print(f"\nDone. Deleted {deleted:,} duplicate rows.")

    with conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM reviews")
        final_count = cur.fetchone()[0]
    print(f"reviews table now has {final_count:,} rows.")

    conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
