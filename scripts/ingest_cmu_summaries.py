"""
Ingest CMU Book Summary Dataset plot summaries into the books table.

The CMU dataset provides plot summaries for ~16.5k books. It has no ISBN
fields, so we cannot key off isbn13. Instead, we match against existing rows
in the books table by (normalized_title, normalized_first_author) and UPDATE
plot_summary for rows that match. Rows with no match in the DB are inserted
as new book records with source=cmu_summaries.

Source file: data/raw/booksummaries/booksummaries.txt (tab-separated)
             Extracted by download_datasets.py.
Target table: books (DB_1)

Column layout (7 tab-separated fields, 0-indexed):
    0  wikipedia_id      - integer
    1  freebase_id       - Freebase MID string (e.g. /m/0hhy)
    2  title             - book title
    3  author            - single author name (often just surname or full name)
    4  pub_date          - publication date (YYYY-MM-DD, YYYY, or blank)
    5  genres            - JSON object {"freebase_id": "genre_name", ...} or blank
    6  summary           - plot summary (free text, may be very long)

Key quirks:
- Genres are JSON objects mapping Freebase IDs to names. We extract only the
  names. The CMU genre vocabulary is broader and noisier than Goodreads.
- Author is a single field, sometimes just "Orwell" or "Rowling". We store it
  as a one-element list so it fits the authors: text[] column.
- Pub_date may be a full ISO date, a year only, or blank.
- The file encoding is UTF-8 but occasionally has raw HTML entities like
  &amp; or &#39; that we unescape.

Matching strategy (CMU has no ISBN):
    1. Normalize both sides: lowercase, strip punctuation, collapse whitespace.
    2. SELECT isbn13, title, authors FROM books WHERE source = 'goodreads_bbe'
       and build an in-memory index: {normalized_key -> isbn13}.
       Normalized key = norm(title) + "|" + norm(authors[0] if any else "").
    3. For each CMU row: look up normalized key. If found, UPDATE plot_summary
       and (if blank) publication info. If not found, INSERT as new row.

This means the CMU ingestor must run AFTER the Zenodo ingestor.

Usage:
    uv run python scripts/ingest_cmu_summaries.py
    uv run python scripts/ingest_cmu_summaries.py --dry-run
    uv run python scripts/ingest_cmu_summaries.py --limit 200
"""

from __future__ import annotations

import argparse
import html
import json
import re
import sys
from datetime import date
from pathlib import Path
from typing import Optional

import psycopg2
from tqdm import tqdm

import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from lib.db import BULK_PAGE_SIZE, get_connection
from lib.models import BookRow, BookSource


DATA_FILE = Path("data/raw/booksummaries/booksummaries.txt")

BATCH_SIZE = 500


# --- Text normalization for fuzzy matching ---

_STRIP_PUNCT = re.compile(r"[^\w\s]")
_COLLAPSE_SPACE = re.compile(r"\s+")
# Common title words that add noise in matching.
_ARTICLES = re.compile(r"^(the|a|an)\s+", re.IGNORECASE)


def normalize_for_match(text: str) -> str:
    """Lowercase, remove articles, strip punctuation, collapse whitespace."""
    text = text.lower().strip()
    text = _ARTICLES.sub("", text)
    text = html.unescape(text)
    text = _STRIP_PUNCT.sub("", text)
    text = _COLLAPSE_SPACE.sub(" ", text).strip()
    return text


def match_key(title: str, first_author: str) -> str:
    return normalize_for_match(title) + "|" + normalize_for_match(first_author)


# --- Parsing helpers ---

def parse_genres_json(raw: str) -> list[str]:
    """Extract genre name strings from the Freebase JSON dict."""
    raw = raw.strip()
    if not raw:
        return []
    try:
        obj = json.loads(raw)
        # Values are genre names; keys are Freebase MIDs which we discard.
        return [str(v).strip() for v in obj.values() if v]
    except (json.JSONDecodeError, ValueError):
        return []


def parse_pub_date(raw: str) -> Optional[date]:
    """Parse YYYY-MM-DD, YYYY-MM, or YYYY date strings."""
    raw = raw.strip()
    if not raw:
        return None
    try:
        if len(raw) == 10 and raw[4] == "-":
            from datetime import datetime
            return datetime.strptime(raw, "%Y-%m-%d").date()
        if len(raw) == 7 and raw[4] == "-":
            from datetime import datetime
            return datetime.strptime(raw, "%Y-%m").date()
        year = int(raw[:4])
        if 1000 <= year <= 2100:
            return date(year, 1, 1)
    except (ValueError, TypeError):
        pass
    return None


def unescape_summary(text: str) -> str:
    """Remove HTML entities (e.g. &amp; -> &) from plot summaries."""
    return html.unescape(text).strip()


# --- Match-index builder ---

def build_match_index(conn) -> dict[str, str]:
    """Load all existing book rows and return a key -> isbn13 lookup dict.

    We only index rows that have an isbn13 (so we can update them cleanly).
    Rows without isbn13 would need a different dedup strategy.
    """
    index: dict[str, str] = {}
    with conn.cursor() as cur:
        cur.execute("SELECT isbn13, title, authors FROM books WHERE isbn13 IS NOT NULL")
        for isbn13, title, authors in cur.fetchall():
            first_author = (authors[0] if authors else "") if authors else ""
            key = match_key(title or "", first_author)
            index[key] = isbn13
    return index


# --- Bulk update helper ---

def bulk_update_plot_summaries(conn, updates: list[tuple[str, str]]) -> int:
    """UPDATE plot_summary for (isbn13, summary) pairs.

    Uses executemany since we're updating individual rows; this is not as
    fast as execute_values but acceptable for a dataset of 16k rows.
    """
    if not updates:
        return 0
    with conn.cursor() as cur:
        # Only update if plot_summary is currently NULL - don't overwrite if
        # another source already provided a better summary.
        cur.executemany(
            "UPDATE books SET plot_summary = %s WHERE isbn13 = %s AND plot_summary IS NULL",
            [(summary, isbn13) for isbn13, summary in updates],
        )
    conn.commit()
    return len(updates)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dry-run", action="store_true", help="Parse and match without writing to DB.")
    parser.add_argument("--limit", type=int, default=None, help="Process at most N rows.")
    args = parser.parse_args()

    if not DATA_FILE.exists():
        print(
            f"ERROR: {DATA_FILE} not found.\n"
            "Run: uv run python scripts/download_datasets.py",
            file=sys.stderr,
        )
        return 1

    conn = None if args.dry_run else get_connection()

    if not args.dry_run:
        print("Building match index from existing books table...")
        match_index = build_match_index(conn)
        print(f"  {len(match_index):,} books indexed for matching.")
    else:
        match_index = {}

    total = 0
    matched = 0
    unmatched_new_inserts = 0
    parse_errors = 0

    pending_updates: list[tuple[str, str]] = []   # (isbn13, summary)
    pending_inserts: list[BookRow] = []

    with open(DATA_FILE, encoding="utf-8", errors="replace") as f:
        line_iter = enumerate(f, start=1)
        if args.limit is not None:
            line_iter = ((n, l) for n, l in line_iter if n <= args.limit)

        for lineno, line in tqdm(line_iter, desc="Parsing CMU rows", unit="row"):
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 7:
                # Some lines have fewer fields (missing genres or summary).
                fields += [""] * (7 - len(fields))

            _, _, title_raw, author_raw, pub_date_raw, genres_raw, summary_raw = (
                fields[0], fields[1], fields[2], fields[3],
                fields[4], fields[5], fields[6],
            )

            title = html.unescape(title_raw.strip())
            author = html.unescape(author_raw.strip())
            summary = unescape_summary(summary_raw)

            if not title or not summary:
                # A row without a title or summary has nothing to contribute.
                parse_errors += 1
                continue

            total += 1

            # Try to match against an existing Zenodo row.
            key = match_key(title, author)
            isbn13 = match_index.get(key)

            if isbn13:
                matched += 1
                if not args.dry_run:
                    pending_updates.append((isbn13, summary))
            else:
                # No match - insert as a new CMU-only row. It won't have ISBN;
                # the ISBNdb enrichment pass may fill it in later.
                unmatched_new_inserts += 1
                if not args.dry_run:
                    genres = parse_genres_json(genres_raw)
                    pub_date = parse_pub_date(pub_date_raw)
                    pending_inserts.append(
                        BookRow(
                            isbn13=None,
                            title=title,
                            authors=[author] if author else [],
                            publish_date=pub_date,
                            genres=genres,
                            plot_summary=summary,
                            source=BookSource.CMU_SUMMARIES,
                        )
                    )

            # Flush updates in batches to avoid unbounded memory growth.
            if len(pending_updates) >= BULK_PAGE_SIZE and not args.dry_run:
                bulk_update_plot_summaries(conn, pending_updates)
                pending_updates.clear()

            if len(pending_inserts) >= BULK_PAGE_SIZE and not args.dry_run:
                from lib.db import bulk_insert_books
                bulk_insert_books(conn, pending_inserts, on_conflict="DO NOTHING")
                pending_inserts.clear()

    # Flush remaining.
    if not args.dry_run:
        if pending_updates:
            bulk_update_plot_summaries(conn, pending_updates)
        if pending_inserts:
            from lib.db import bulk_insert_books
            bulk_insert_books(conn, pending_inserts, on_conflict="DO NOTHING")
        conn.close()

    print()
    print(f"Total CMU rows processed: {total:,}")
    print(f"  Matched to Zenodo rows: {matched:,}  (plot_summary updated)")
    print(f"  Unmatched (new inserts): {unmatched_new_inserts:,}")
    print(f"  Skipped (no title/summary): {parse_errors:,}")
    if args.dry_run:
        print("Dry run - nothing written.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
