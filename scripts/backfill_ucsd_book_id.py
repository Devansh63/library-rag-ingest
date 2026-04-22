"""
One-time backfill: populate books.ucsd_book_id for existing UCSD rows.

This script is needed because ucsd_book_id was added after the initial UCSD
ingest. Without it, the reviews pass cannot link reviews for no-isbn books
(books.ucsd_book_id is the bridge from UCSD's internal book_id to books.id).

What it does:
  1. Queries the DB for all no-isbn UCSD books (isbn13 IS NULL, source = 'ucsd_graph')
     and builds a title+author match index in memory.
  2. Streams the UCSD books file once, matching each book by title+author.
  3. Batch-UPDATEs ucsd_book_id for all matched rows.

After this script finishes, run:
    uv run python scripts/ingest_ucsd_graph.py --reviews-only

That will stream the reviews file and attach reviews to the previously skipped
no-isbn books via the reviews.book_id FK.

Usage:
    uv run python scripts/backfill_ucsd_book_id.py
    uv run python scripts/backfill_ucsd_book_id.py --dry-run
"""

from __future__ import annotations

import argparse
import gzip
import html
import json
import re
import sys
from pathlib import Path

from psycopg2.extras import execute_values
from tqdm import tqdm

import pathlib; sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from lib.db import get_connection, BULK_PAGE_SIZE
from lib.db import ensure_ucsd_book_id_column, ensure_review_book_id_column


BOOKS_FILE = Path("data/raw/ucsd_goodreads_books.json.gz")

_STRIP_PUNCT = re.compile(r"[^\w\s]")
_ARTICLES = re.compile(r"^(the|a|an)\s+", re.IGNORECASE)
_HTML_TAG_RE = re.compile(r"<[^>]+>")


def normalize(text: str) -> str:
    text = text.lower().strip()
    text = _ARTICLES.sub("", text)
    text = html.unescape(text)
    text = _STRIP_PUNCT.sub("", text)
    return re.sub(r"\s+", " ", text).strip()


def clean_html(text: str) -> str:
    text = _HTML_TAG_RE.sub(" ", text)
    text = html.unescape(text)
    return re.sub(r"\s+", " ", text).strip()


def build_db_match_index(conn) -> dict[str, int]:
    """Load all no-isbn UCSD books from DB into a title+author -> books.id map.

    We only target no-isbn books because isbn books already get ucsd_book_id
    set via _flush_book_updates when ingest_ucsd_graph.py is re-run.
    """
    index: dict[str, int] = {}
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT id, title, authors
            FROM books
            WHERE isbn13 IS NULL
              AND source = 'ucsd_graph'
              AND ucsd_book_id IS NULL
            """
        )
        rows = cur.fetchall()

    print(f"  {len(rows):,} no-isbn UCSD books without ucsd_book_id loaded from DB.")
    for db_id, title, authors in rows:
        first_author = (authors[0] if authors else "") if authors else ""
        key = normalize(title or "") + "|" + normalize(first_author)
        if key not in index:
            index[key] = db_id
    return index


def flush_updates(conn, updates: list[tuple], dry_run: bool) -> int:
    """Batch-UPDATE ucsd_book_id for a list of (db_id, ucsd_book_id) pairs."""
    if not updates or dry_run:
        return 0
    sql = """
        UPDATE books AS b
        SET ucsd_book_id = v.ucsd_id
        FROM (VALUES %s) AS v(db_id, ucsd_id)
        WHERE b.id = v.db_id::integer
          AND b.ucsd_book_id IS NULL
    """
    template = "(%s::integer, %s::text)"
    with conn.cursor() as cur:
        execute_values(cur, sql, updates, template=template, page_size=BULK_PAGE_SIZE)
    conn.commit()
    return len(updates)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dry-run", action="store_true",
                        help="Scan and match but write nothing to the DB.")
    args = parser.parse_args()

    if not BOOKS_FILE.exists():
        print(f"ERROR: {BOOKS_FILE} not found.\n"
              "Run: uv run python scripts/download_datasets.py --ucsd",
              file=sys.stderr)
        return 1

    conn = get_connection()
    ensure_ucsd_book_id_column(conn)
    ensure_review_book_id_column(conn)

    print("Loading no-isbn UCSD books from DB...")
    db_index = build_db_match_index(conn)

    if not db_index:
        print("Nothing to backfill - all no-isbn UCSD books already have ucsd_book_id set.")
        conn.close()
        return 0

    print(f"\nStreaming {BOOKS_FILE.name} to match ucsd_book_ids...")

    matched = 0
    not_in_db = 0
    already_set = 0
    pending: list[tuple] = []

    with gzip.open(BOOKS_FILE, "rt", encoding="utf-8", errors="replace") as f:
        for line in tqdm(f, desc="UCSD books", unit="row"):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue

            # Only process books with no isbn13 - isbn books are handled by
            # ingest_ucsd_graph.py's _flush_book_updates on the next run.
            isbn13_raw = obj.get("isbn13") or obj.get("isbn") or ""
            if isbn13_raw.strip():
                continue

            ucsd_id = str(obj.get("book_id", ""))
            if not ucsd_id:
                continue

            title_raw = clean_html(obj.get("title", "").strip())
            title_clean = clean_html(
                obj.get("title_without_series", "").strip()
            ) or title_raw

            authors_list = obj.get("authors", [])
            first_author = ""
            if authors_list and isinstance(authors_list[0], dict):
                first_author = str(authors_list[0].get("name", "")).strip()

            key = normalize(title_clean) + "|" + normalize(first_author)
            db_id = db_index.get(key)

            if db_id is None:
                not_in_db += 1
                continue

            matched += 1
            pending.append((db_id, ucsd_id))
            # Remove from index so we don't match the same DB row twice.
            del db_index[key]

            if len(pending) >= BULK_PAGE_SIZE:
                flush_updates(conn, pending, args.dry_run)
                pending.clear()

    # Final flush.
    flush_updates(conn, pending, args.dry_run)

    conn.close()

    print(f"\nBackfill complete.")
    print(f"  Matched and updated:  {matched:,}")
    print(f"  In file, not in DB:   {not_in_db:,}")
    print(f"  DB books unmatched:   {len(db_index):,} (title changed or file mismatch)")
    if args.dry_run:
        print("  (dry run - nothing written)")
    else:
        print(f"\nNext step: recover reviews for these books by running:")
        print(f"  uv run python scripts/ingest_ucsd_graph.py --reviews-only")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
