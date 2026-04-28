"""
Select the best N books for embedding and mark them with 'EMBED_QUEUED' in
cleaning_flags. The Mac mini (or any other machine) then picks up exactly
these books via embed_sample.py --queued.

WHY MARK INSTEAD OF RANDOM:
  Random sampling gives us a representative slice but ignores quality signals.
  Books with more reviews produce better KMeans clusters. Books from trusted
  sources (goodreads_bbe, cmu_summaries) have richer metadata. Marking lets
  us pick the best candidates deliberately and hand them to any machine.

SCORING FORMULA (higher = better candidate for embedding):
  source_score:
    goodreads_bbe or cmu_summaries -> 3  (trusted, rich metadata)
    ucsd_graph CLEANED             -> 1  (ISBNdb-enriched, decent)
  review_score: log10(review_count + 1)  (diminishing returns past ~100 reviews)
  description_score:
    has plot_summary    -> 2  (richest - scholarly summary)
    has synopsis        -> 1
    has short_desc only -> 0
  final = source_score * 2 + review_score * 3 + description_score

Usage:
    uv run python scripts/embedding/mark_embed_queue.py --limit 20000 --dry-run
    uv run python scripts/embedding/mark_embed_queue.py --limit 20000
    uv run python scripts/embedding/mark_embed_queue.py --clear   # remove all marks
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent.parent / ".env")

from lib.db import get_connection

FLAG = "EMBED_QUEUED"
DEFAULT_LIMIT = 20_000


def clear_queue(conn) -> int:
    """Remove EMBED_QUEUED flag from all books."""
    with conn.cursor() as cur:
        cur.execute("""
            UPDATE books
            SET cleaning_flags = array_remove(cleaning_flags, %s)
            WHERE %s = ANY(cleaning_flags)
        """, (FLAG, FLAG))
        count = cur.rowcount
    conn.commit()
    return count


def count_queued(conn) -> int:
    with conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM books WHERE %s = ANY(cleaning_flags)", (FLAG,))
        return cur.fetchone()[0]


def mark_best_books(conn, limit: int, dry_run: bool) -> int:
    """Score and mark the top `limit` unembedded quality books."""

    print(f"Scoring unembedded quality books (selecting top {limit:,})...")
    with conn.cursor() as cur:
        cur.execute("""
            WITH
            -- Pre-aggregate review counts (CTE avoids correlated subquery on 11M rows)
            isbn_counts AS (
                SELECT isbn13, COUNT(*) AS cnt
                FROM reviews
                WHERE isbn13 IS NOT NULL
                GROUP BY isbn13
            ),
            id_counts AS (
                SELECT book_id, COUNT(*) AS cnt
                FROM reviews
                WHERE book_id IS NOT NULL
                GROUP BY book_id
            ),
            candidates AS (
                SELECT
                    b.id,
                    b.title,
                    b.source,
                    -- Source quality score
                    CASE
                        WHEN b.source IN ('goodreads_bbe', 'cmu_summaries') THEN 3
                        ELSE 1
                    END AS source_score,
                    -- Review count score: log10(count + 1), capped signal
                    LOG(COALESCE(ic.cnt, 0) + COALESCE(idc.cnt, 0) + 1) AS review_score,
                    -- Description richness score
                    CASE
                        WHEN b.plot_summary IS NOT NULL THEN 2
                        WHEN b.synopsis IS NOT NULL THEN 1
                        ELSE 0
                    END AS desc_score,
                    COALESCE(ic.cnt, 0) + COALESCE(idc.cnt, 0) AS review_count
                FROM books b
                LEFT JOIN isbn_counts  ic  ON ic.isbn13   = b.isbn13
                LEFT JOIN id_counts    idc ON idc.book_id = b.id
                WHERE
                    -- Quality sources only
                    (
                        b.source IN ('goodreads_bbe', 'cmu_summaries')
                        OR (b.source = 'ucsd_graph' AND 'CLEANED' = ANY(b.cleaning_flags))
                    )
                    -- Must have a description
                    AND (
                        b.synopsis          IS NOT NULL OR
                        b.short_description IS NOT NULL OR
                        b.plot_summary      IS NOT NULL
                    )
                    -- Must have enough reviews for meaningful KMeans
                    AND (COALESCE(ic.cnt, 0) + COALESCE(idc.cnt, 0)) >= 30
                    -- Skip already embedded
                    AND b.metadata_embedding IS NULL
                    -- Skip already queued
                    AND NOT (%s = ANY(COALESCE(b.cleaning_flags, ARRAY[]::text[])))
            )
            SELECT
                id,
                title,
                source,
                review_count,
                -- Weighted composite score
                (source_score * 2.0 + review_score * 3.0 + desc_score) AS score
            FROM candidates
            ORDER BY score DESC
            LIMIT %s
        """, (FLAG, limit))

        rows = cur.fetchall()

    if not rows:
        print("No qualifying unembedded books found.")
        return 0

    print(f"  Found {len(rows):,} books to mark.")
    print(f"\n  Score distribution (top 5 / bottom 5):")
    for r in rows[:5]:
        print(f"    [{r[4]:.2f}] {r[1][:55]}  source={r[2]}  reviews={r[3]:,}")
    print("    ...")
    for r in rows[-5:]:
        print(f"    [{r[4]:.2f}] {r[1][:55]}  source={r[2]}  reviews={r[3]:,}")

    if dry_run:
        print(f"\n[dry-run] Would mark {len(rows):,} books with '{FLAG}'. No changes made.")
        return len(rows)

    # Mark them
    book_ids = [r[0] for r in rows]
    with conn.cursor() as cur:
        cur.execute("""
            UPDATE books
            SET cleaning_flags = array_append(COALESCE(cleaning_flags, ARRAY[]::text[]), %s)
            WHERE id = ANY(%s)
              AND NOT (%s = ANY(COALESCE(cleaning_flags, ARRAY[]::text[])))
        """, (FLAG, book_ids, FLAG))
        marked = cur.rowcount
    conn.commit()

    print(f"\n  Marked {marked:,} books with '{FLAG}'.")
    return marked


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--limit", type=int, default=DEFAULT_LIMIT,
                        help=f"Number of books to mark (default {DEFAULT_LIMIT:,}).")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be marked without writing.")
    parser.add_argument("--clear", action="store_true",
                        help="Remove all EMBED_QUEUED marks and exit.")
    args = parser.parse_args()

    conn = get_connection()

    if args.clear:
        removed = clear_queue(conn)
        print(f"Removed '{FLAG}' from {removed:,} books.")
        conn.close()
        return

    already_queued = count_queued(conn)
    if already_queued:
        print(f"Note: {already_queued:,} books already marked '{FLAG}' (will be skipped).")

    mark_best_books(conn, args.limit, dry_run=args.dry_run)
    conn.close()


if __name__ == "__main__":
    main()
