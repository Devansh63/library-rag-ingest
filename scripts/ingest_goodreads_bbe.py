"""
Ingest the GoodReads Best Books Ever dataset (Zenodo) into the books table.

This is the first and primary ingestion pass. It establishes the isbn13 keys
that all subsequent passes (CMU, UCSD, ISBNdb) key off of. Rows without an
isbn13 are still inserted but will not participate in ON CONFLICT dedup logic.

Source file: data/raw/goodreads_bbe.csv (downloaded by download_datasets.py)
Target table: books (DB_1)

Key quirks of this dataset:
- `isbn` is almost always ISBN-13 already (no hyphens).
- `genres` and `awards` are stored as Python list literals, e.g.
  "['Young Adult', 'Fiction']". We use ast.literal_eval to parse them.
- `author` is a single string that sometimes includes roles in parentheses
  like "J.K. Rowling, Mary GrandPré (Illustrator)". We split on commas,
  strip parenthetical roles, and deduplicate.
- Dates are in MM/DD/YY format (e.g. "09/14/08"). We convert to Python date.
- `language` is a full English name ("English", "Spanish") not an ISO code.
  We normalize to lowercase 2-letter codes where we can; leave others as-is.

Usage:
    uv run python scripts/ingest_goodreads_bbe.py
    uv run python scripts/ingest_goodreads_bbe.py --dry-run   # parse only
    uv run python scripts/ingest_goodreads_bbe.py --limit 500 # test subset
"""

from __future__ import annotations

import argparse
import ast
import csv
import re
import sys
from datetime import date
from pathlib import Path
from typing import Optional

from tqdm import tqdm

# Ensure project root is on sys.path before importing from lib/.
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from lib.db import bulk_insert_books, get_connection
from lib.isbn import normalize_isbn13
from lib.models import BookRow, BookSource


DATA_FILE = Path("data/raw/goodreads_bbe.csv")

# Rough mapping from full English language names to ISO 639-1 codes.
# Extend as you encounter new values in the dataset.
LANGUAGE_MAP = {
    "english": "en",
    "spanish": "es",
    "french": "fr",
    "german": "de",
    "portuguese": "pt",
    "italian": "it",
    "dutch": "nl",
    "japanese": "ja",
    "chinese": "zh",
    "russian": "ru",
    "arabic": "ar",
    "korean": "ko",
    "swedish": "sv",
    "polish": "pl",
    "turkish": "tr",
}

# Parenthetical role tags we strip from author strings (case-insensitive).
_ROLE_PATTERN = re.compile(r"\s*\([^)]*\)", re.IGNORECASE)

# Only insert in batches of this size to keep memory usage predictable.
BATCH_SIZE = 500


def parse_python_list(raw: str) -> list[str]:
    """Parse a stringified Python list like "['a', 'b']" into a real list.

    Returns an empty list if the string is blank or can't be parsed.
    We use ast.literal_eval because the field is a literal from a Python
    script, not JSON. json.loads would choke on single quotes.
    """
    raw = raw.strip()
    if not raw or raw in ("[]", "None"):
        return []
    try:
        result = ast.literal_eval(raw)
        if isinstance(result, list):
            # Filter out any non-string junk and strip each value.
            return [str(item).strip() for item in result if item]
        return []
    except (ValueError, SyntaxError):
        return []


def parse_authors(raw: str) -> list[str]:
    """Split a multi-author string into a clean list.

    Input like "J.K. Rowling, Mary GrandPré (Illustrator)" becomes
    ["J.K. Rowling", "Mary GrandPré"].
    """
    if not raw:
        return []
    # Strip parenthetical roles (Illustrator, Translator, Editor, etc.).
    cleaned = _ROLE_PATTERN.sub("", raw)
    parts = [p.strip() for p in cleaned.split(",") if p.strip()]
    # Deduplicate while preserving order (the same author listed twice is a
    # data quality issue in the source that we fix silently here).
    seen: set[str] = set()
    result: list[str] = []
    for part in parts:
        if part not in seen:
            seen.add(part)
            result.append(part)
    return result


def parse_date(raw: str) -> Optional[date]:
    """Parse a date string in MM/DD/YY format, or return None.

    The dataset uses two-digit years (e.g. "09/14/08" means 2008-09-14).
    Python's %y directive maps 00-68 to 2000-2068 and 69-99 to 1969-1999,
    which is correct for publication dates in this dataset.
    """
    raw = raw.strip()
    if not raw:
        return None
    try:
        from datetime import datetime
        return datetime.strptime(raw, "%m/%d/%y").date()
    except ValueError:
        # Try year-only format as a fallback (some rows have just "2008").
        try:
            year = int(raw[:4])
            if 1000 <= year <= 2100:
                return date(year, 1, 1)
        except (ValueError, TypeError):
            pass
    return None


def normalize_language(raw: str) -> Optional[str]:
    """Map a full language name to an ISO 639-1 code, or None if unknown.

    The DB column is varchar(10), intended for short language codes.
    We only return values we know are valid ISO 639-1 codes from LANGUAGE_MAP.
    Unmapped languages return None (DB will use the default 'en'); this is
    conservative but avoids truncation errors on long language names.
    """
    if not raw:
        return None
    return LANGUAGE_MAP.get(raw.strip().lower())


def row_to_book(csv_row: dict) -> Optional[BookRow]:
    """Convert one CSV row dict to a BookRow. Returns None if the row is unusable."""
    title = csv_row.get("title", "").strip()
    if not title:
        # A book without a title is useless in the catalog.
        return None

    isbn13 = normalize_isbn13(csv_row.get("isbn"))

    rating_raw = csv_row.get("rating", "").strip()
    try:
        goodreads_rating = float(rating_raw) if rating_raw else None
    except ValueError:
        goodreads_rating = None

    num_ratings_raw = csv_row.get("numRatings", "").strip()
    try:
        num_ratings = int(num_ratings_raw) if num_ratings_raw else None
    except ValueError:
        num_ratings = None

    pages_raw = csv_row.get("pages", "").strip()
    try:
        pages = int(pages_raw) if pages_raw else None
        pages = pages if pages and pages > 0 else None
    except ValueError:
        pages = None

    return BookRow(
        isbn13=isbn13,
        isbn10=None,  # BBE only provides ISBN-13.
        title=title,
        authors=parse_authors(csv_row.get("author", "")),
        publisher=csv_row.get("publisher", "").strip() or None,
        publish_date=parse_date(csv_row.get("publishDate", "")),
        first_publish_date=parse_date(csv_row.get("firstPublishDate", "")),
        genres=parse_python_list(csv_row.get("genres", "")),
        subjects=[],  # BBE doesn't have subjects; ISBNdb enrichment will fill these.
        language=normalize_language(csv_row.get("language", "")),
        pages=pages,
        edition=csv_row.get("edition", "").strip() or None,
        series=csv_row.get("series", "").strip() or None,
        short_description=None,  # No separate short description in BBE.
        synopsis=csv_row.get("description", "").strip() or None,
        plot_summary=None,       # Filled in by CMU ingestor.
        cover_image_url=csv_row.get("coverImg", "").strip() or None,
        goodreads_rating=goodreads_rating,
        num_ratings=num_ratings,
        awards=parse_python_list(csv_row.get("awards", "")),
        source=BookSource.GOODREADS_BBE,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dry-run", action="store_true", help="Parse and validate rows without inserting.")
    parser.add_argument("--limit", type=int, default=None, help="Only process the first N rows (for testing).")
    args = parser.parse_args()

    if not DATA_FILE.exists():
        print(
            f"ERROR: {DATA_FILE} not found. Run scripts/download_datasets.py first.",
            file=sys.stderr,
        )
        return 1

    conn = None if args.dry_run else get_connection()

    total_rows = 0
    valid_rows = 0
    skipped_no_title = 0
    skipped_invalid = 0
    inserted = 0

    batch: list[BookRow] = []

    try:
        with open(DATA_FILE, newline="", encoding="utf-8", errors="replace") as f:
            reader = csv.DictReader(f)
            row_iter = reader if args.limit is None else (r for i, r in enumerate(reader) if i < args.limit)

            for csv_row in tqdm(row_iter, desc="Parsing rows", unit="row"):
                total_rows += 1
                try:
                    book = row_to_book(csv_row)
                except Exception as exc:
                    # Don't crash on a bad row; log and skip.
                    print(f"  WARN: row {total_rows} failed validation: {exc}", file=sys.stderr)
                    skipped_invalid += 1
                    continue

                if book is None:
                    skipped_no_title += 1
                    continue

                valid_rows += 1
                batch.append(book)

                if len(batch) >= BATCH_SIZE and not args.dry_run:
                    inserted += bulk_insert_books(conn, batch)
                    batch.clear()

        # Flush remaining rows.
        if batch and not args.dry_run:
            inserted += bulk_insert_books(conn, batch)
            batch.clear()

    finally:
        if conn:
            conn.close()

    print()
    print(f"Total rows read:     {total_rows:,}")
    print(f"Valid BookRows:      {valid_rows:,}")
    print(f"Skipped (no title):  {skipped_no_title:,}")
    print(f"Skipped (invalid):   {skipped_invalid:,}")
    if not args.dry_run:
        print(f"Inserted (approx):   {inserted:,}")
    else:
        print("Dry run - nothing inserted.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
