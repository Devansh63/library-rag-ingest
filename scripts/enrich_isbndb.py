"""
Enrich the books table using the ISBNdb API (Basic plan).

Looks up books by ISBN13 in batches of 100 using POST /books, and:
  1. Fills missing book fields (authors, synopsis, etc.)
  2. Stores editorial/professional reviews separately from user reviews
  3. Treats ISBNdb as authoritative for catalog fields (isbn10, language,
     publisher, pages) - these are OVERWRITTEN, not just filled when NULL
  4. Merges subjects (union of existing + ISBNdb, no duplicates)

FIELDS AND UPDATE STRATEGY:
  Overwrite (ISBNdb is authoritative catalog source):
    - isbn10         : verified ISBN-10
    - publisher      : canonical publisher name
    - language       : authoritative ISO language code
    - pages          : accurate page count

  Fill NULL only (existing data may be richer):
    - authors        : Zenodo/CMU may have fuller author lists
    - synopsis       : existing might be longer
    - short_description (excerpt): fallback when synopsis absent
    - cover_image_url
    - edition
    - publish_date

  Merge (union, no replace):
    - subjects       : append ISBNdb subjects to existing, deduplicated

  New column (always fill if available):
    - dewey_decimal  : standardized library classification (new column)

EDITORIAL REVIEWS:
  ISBNdb returns professional reviews (Kirkus, Publishers Weekly, etc.).
  These are stored in the reviews table with review_type='editorial'.
  User reviews from UCSD have review_type='user' (default).
  - user_id holds the publication/source name for editorial rows
  - rating is NULL (editorial reviews have no star rating)
  Query: WHERE isbn13 = ? AND review_type = 'editorial'

CLEANED FLAG:
  When enrichment fills at least one meaningful field (authors or synopsis),
  'CLEANED' is added to cleaning_flags.
  Query enriched books: WHERE cleaning_flags @> ARRAY['CLEANED']

PRIORITY ORDER:
  1. missing_description + isbn13 (worst embedding quality, 152K books)
  2. missing_author + isbn13 (all remaining UCSD books)

DAILY QUOTA:
  Basic plan: 5,000 calls/day. Each ISBN = 1 call.
  50 batches/day x 100 ISBNs = 5,000 ISBNs/day.
  GET /key checked before starting for exact remaining count.

Usage:
  uv run python scripts/enrich_isbndb.py
  uv run python scripts/enrich_isbndb.py --dry-run
  uv run python scripts/enrich_isbndb.py --limit 300
  uv run python scripts/enrich_isbndb.py --priority-only
  uv run python scripts/enrich_isbndb.py --daily-buffer 200
  uv run python scripts/enrich_isbndb.py --review-boost --max-reviews 60 --limit 5000
  uv run python scripts/enrich_isbndb.py --embed-prep --limit 5000
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from datetime import date
from typing import Optional

import requests
from psycopg2.extras import execute_values
from tqdm import tqdm

import pathlib; sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from lib.db import get_connection, bulk_insert_reviews
from lib.models import ReviewRow, ReviewSource, ReviewType
from dotenv import load_dotenv
import os

load_dotenv()

ISBNDB_API_KEY = os.environ.get("ISBNDB_API_KEY", "")
BASE_URL = "https://api2.isbndb.com"

# Stop when daily remaining drops to this level - prevents overage charges.
DEFAULT_DAILY_BUFFER = 100

# Basic plan: 100 ISBNs per bulk request.
BULK_BATCH_SIZE = 100

# Synopsis must be at least this long to count as "filled".
MIN_DESCRIPTION_LEN = 50

# DB flush size for book updates and editorial reviews.
DB_FLUSH_SIZE = 50


# ---------------------------------------------------------------------------
# Rate limit tracking
# ---------------------------------------------------------------------------

_RL_PATTERN = re.compile(r'"(\w+)";r=(\d+);t=(\d+)')


def parse_ratelimit(header: str) -> dict[str, dict]:
    """Parse ratelimit header: '"rate";r=59;t=1, "daily";r=4990;t=56800'"""
    result = {}
    for match in _RL_PATTERN.finditer(header):
        result[match.group(1)] = {
            "remaining": int(match.group(2)),
            "reset_in": int(match.group(3)),
        }
    return result


class RateLimiter:
    def __init__(self, daily_buffer: int = DEFAULT_DAILY_BUFFER):
        self.daily_buffer = daily_buffer
        self.rate_remaining = 60
        self.rate_reset_in = 1
        self.daily_remaining = 5000
        self.daily_reset_in = 86400

    def update(self, response: requests.Response) -> None:
        header = response.headers.get("ratelimit", "")
        if not header:
            return
        parsed = parse_ratelimit(header)
        if "rate" in parsed:
            self.rate_remaining = parsed["rate"]["remaining"]
            self.rate_reset_in = parsed["rate"]["reset_in"]
        if "daily" in parsed:
            self.daily_remaining = parsed["daily"]["remaining"]
            self.daily_reset_in = parsed["daily"]["reset_in"]

    def daily_exhausted(self, batch_size: int = BULK_BATCH_SIZE) -> bool:
        return self.daily_remaining - batch_size < self.daily_buffer

    def wait_if_needed(self) -> None:
        if self.rate_remaining <= 1:
            sleep_secs = max(self.rate_reset_in + 0.5, 1.0)
            tqdm.write(f"  [rate limiter] window exhausted, sleeping {sleep_secs:.1f}s...")
            time.sleep(sleep_secs)
        else:
            time.sleep(1.05)


# ---------------------------------------------------------------------------
# ISBNdb API helpers
# ---------------------------------------------------------------------------

def _headers() -> dict[str, str]:
    return {"Authorization": ISBNDB_API_KEY, "Content-Type": "application/json"}


def preflight_quota_check() -> int:
    """GET /key - returns exact remaining quota before spending any calls."""
    try:
        resp = requests.get(f"{BASE_URL}/key", headers=_headers(), timeout=10)
        if resp.status_code == 200:
            plan = resp.json().get("plan_limit", {})
            left = plan.get("left", -1)
            total = plan.get("total", -1)
            spent = plan.get("spent", -1)
            print(f"  API quota: {left:,} remaining / {total:,} total ({spent:,} spent today)")
            return left
        print(f"  [preflight] HTTP {resp.status_code}: {resp.text[:80]}")
        return -1
    except requests.RequestException as exc:
        print(f"  [preflight] network error: {exc}")
        return -1


def lookup_isbns_bulk(isbn_list: list[str], rate_limiter: RateLimiter) -> dict[str, dict]:
    """POST /books with up to 100 ISBNs. Returns {isbn13: book_data} for found books."""
    rate_limiter.wait_if_needed()

    try:
        resp = requests.post(
            f"{BASE_URL}/books",
            headers=_headers(),
            json={"isbns": isbn_list},
            timeout=30,
        )
    except requests.RequestException as exc:
        tqdm.write(f"  [network error] bulk request: {exc}")
        return {}

    rate_limiter.update(resp)

    if resp.status_code == 200:
        books = resp.json().get("data") or []
        result: dict[str, dict] = {}
        for book in books:
            key = book.get("isbn13") or book.get("isbn")
            if key:
                result[key] = book
        return result

    elif resp.status_code == 429:
        tqdm.write("  [429] rate limited, sleeping 65s then retrying...")
        time.sleep(65)
        try:
            resp2 = requests.post(
                f"{BASE_URL}/books", headers=_headers(),
                json={"isbns": isbn_list}, timeout=30,
            )
            rate_limiter.update(resp2)
            if resp2.status_code == 200:
                books = resp2.json().get("data") or []
                return {
                    (b.get("isbn13") or b.get("isbn")): b
                    for b in books if b.get("isbn13") or b.get("isbn")
                }
        except requests.RequestException:
            pass
        return {}

    tqdm.write(f"  [HTTP {resp.status_code}] bulk failed: {resp.text[:120]}")
    return {}


# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------

def fetch_books_to_enrich(
    conn,
    limit: Optional[int],
    priority_only: bool,
    embed_prep: bool = False,
    review_boost: bool = False,
    max_reviews: int = 60,
) -> list[tuple]:
    """Fetch (id, isbn13, cleaning_flags) in priority order.

    Skips books already marked isbndb_not_found or isbndb_checked.

    review_boost mode: targets unembedded goodreads_bbe/cmu_summaries books
    that have isbn13 but fewer reviews than max_reviews. Orders by review
    count DESC so the highest-signal books (closest to the threshold) go
    first. ISBNdb adds editorial reviews which push these books over the
    KMeans minimum. Every API call here is a book that will qualify for
    the embedding queue after marking.

    embed_prep mode: targets ucsd_graph books with 10+ reviews that have
    not yet been cleaned or embedded.

    Standard mode:
      Priority 1: missing_description, Priority 2: missing_author only.
    """
    if review_boost:
        # Pick unembedded goodreads_bbe/cmu books with the most reviews
        # (but below max_reviews cap) that haven't been ISBNdb-checked yet.
        # ISBNdb will add editorial reviews, boosting their review count.
        sql = f"""
            WITH
            isbn_counts AS (
                SELECT isbn13, COUNT(*) AS cnt
                FROM reviews WHERE isbn13 IS NOT NULL GROUP BY isbn13
            ),
            id_counts AS (
                SELECT book_id, COUNT(*) AS cnt
                FROM reviews WHERE book_id IS NOT NULL GROUP BY book_id
            )
            SELECT b.id, b.isbn13, b.cleaning_flags
            FROM books b
            LEFT JOIN isbn_counts ic  ON ic.isbn13   = b.isbn13
            LEFT JOIN id_counts   idc ON idc.book_id = b.id
            WHERE b.source IN ('goodreads_bbe', 'cmu_summaries')
              AND b.isbn13 IS NOT NULL
              AND b.metadata_embedding IS NULL
              AND NOT ('isbndb_checked'   = ANY(COALESCE(b.cleaning_flags, ARRAY[]::text[])))
              AND NOT ('isbndb_not_found' = ANY(COALESCE(b.cleaning_flags, ARRAY[]::text[])))
              AND NOT ('EMBED_QUEUED'     = ANY(COALESCE(b.cleaning_flags, ARRAY[]::text[])))
              AND (b.synopsis IS NOT NULL OR b.short_description IS NOT NULL OR b.plot_summary IS NOT NULL)
              AND (COALESCE(ic.cnt, 0) + COALESCE(idc.cnt, 0)) < %s
            ORDER BY (COALESCE(ic.cnt, 0) + COALESCE(idc.cnt, 0)) DESC
            {"LIMIT " + str(limit) if limit else ""}
        """
        with conn.cursor() as cur:
            cur.execute(sql, (max_reviews,))
            return cur.fetchall()

    if embed_prep:
        sql = f"""
            WITH
            isbn_counts AS (
                SELECT isbn13, COUNT(*) AS cnt
                FROM reviews WHERE isbn13 IS NOT NULL GROUP BY isbn13
            ),
            id_counts AS (
                SELECT book_id, COUNT(*) AS cnt
                FROM reviews WHERE book_id IS NOT NULL GROUP BY book_id
            )
            SELECT b.id, b.isbn13, b.cleaning_flags
            FROM books b
            LEFT JOIN isbn_counts ic  ON ic.isbn13   = b.isbn13
            LEFT JOIN id_counts   idc ON idc.book_id = b.id
            WHERE b.source = 'ucsd_graph'
              AND b.isbn13 IS NOT NULL
              AND b.metadata_embedding IS NULL
              AND NOT ('CLEANED'          = ANY(COALESCE(b.cleaning_flags, ARRAY[]::text[])))
              AND NOT ('isbndb_checked'   = ANY(COALESCE(b.cleaning_flags, ARRAY[]::text[])))
              AND NOT ('isbndb_not_found' = ANY(COALESCE(b.cleaning_flags, ARRAY[]::text[])))
              AND (COALESCE(ic.cnt, 0) + COALESCE(idc.cnt, 0)) >= 10
            ORDER BY (COALESCE(ic.cnt, 0) + COALESCE(idc.cnt, 0)) DESC
            {"LIMIT " + str(limit) if limit else ""}
        """
        with conn.cursor() as cur:
            cur.execute(sql)
            return cur.fetchall()

    flag_filter = (
        "cleaning_flags @> ARRAY['missing_description']"
        if priority_only
        else "(cleaning_flags @> ARRAY['missing_description'] OR cleaning_flags @> ARRAY['missing_author'])"
    )
    sql = f"""
        SELECT id, isbn13, cleaning_flags
        FROM books
        WHERE isbn13 IS NOT NULL
          AND cleaning_flags IS NOT NULL
          AND NOT (cleaning_flags @> ARRAY['isbndb_not_found'])
          AND NOT (cleaning_flags @> ARRAY['isbndb_checked'])
          AND {flag_filter}
        ORDER BY
          CASE WHEN cleaning_flags @> ARRAY['missing_description'] THEN 0 ELSE 1 END,
          num_ratings DESC NULLS LAST
        {"LIMIT " + str(limit) if limit else ""}
    """
    with conn.cursor() as cur:
        cur.execute(sql)
        return cur.fetchall()


def compute_new_flags(
    old_flags: list[str],
    book_data: Optional[dict],
    was_not_found: bool,
    was_enriched: bool,
) -> list[str]:
    """Recompute cleaning_flags after an ISBNdb lookup."""
    flags = set(old_flags or [])

    if was_not_found:
        flags.add("isbndb_not_found")
        return sorted(flags)

    if book_data:
        if book_data.get("authors"):
            flags.discard("missing_author")

        synopsis = (book_data.get("synopsis") or "").strip()
        if synopsis and len(synopsis) >= MIN_DESCRIPTION_LEN:
            flags.discard("missing_description")
            flags.discard("short_description")
        elif synopsis:
            flags.discard("missing_description")
            flags.add("short_description")

    if was_enriched:
        flags.add("CLEANED")

    flags.add("isbndb_checked")
    return sorted(flags) or ["isbndb_checked"]


def extract_editorial_reviews(book_data: dict, isbn13: str) -> list[ReviewRow]:
    """Extract editorial/professional reviews from an ISBNdb book response.

    ISBNdb returns reviews as a list of strings or dicts. We store each one
    as a ReviewRow with review_type='editorial' and user_id set to the
    publication name when available.
    """
    raw_reviews = book_data.get("reviews") or []
    rows = []
    for item in raw_reviews:
        if isinstance(item, str):
            text = item.strip()
            publication = "ISBNdb Editorial"
        elif isinstance(item, dict):
            text = (item.get("review") or item.get("text") or "").strip()
            publication = (item.get("source") or item.get("publication") or "ISBNdb Editorial").strip()
        else:
            continue

        if not text:
            continue

        rows.append(ReviewRow(
            isbn13=isbn13,
            user_id=publication,   # publication name goes in user_id field
            rating=None,           # editorial reviews have no star rating
            review_text=text,
            date_posted=None,
            spoiler_flag=False,
            source=ReviewSource.ISBNDB,
            review_type=ReviewType.EDITORIAL,
        ))
    return rows


def flush_book_updates(conn, updates: list[tuple], dry_run: bool) -> None:
    """Batch-write enrichment results to the books table.

    Each tuple:
        (book_id, authors, synopsis, excerpt, cover_url, publisher,
         pages, language, isbn10, edition, publish_date,
         subjects_to_merge, dewey_decimal, new_cleaning_flags)

    Overwrite strategy (ISBNdb is authoritative for catalog fields):
        - isbn10, publisher, language, pages: ALWAYS overwrite when ISBNdb has a value
        - authors, synopsis, short_description, cover_image_url,
          edition, publish_date: fill NULL only
        - subjects: merged (union of existing + new, deduplicated)
        - dewey_decimal: fill NULL only (new column)
    """
    if not updates or dry_run:
        return

    sql = """
        UPDATE books AS b SET
            -- Fill NULL only: content fields where existing data may be richer
            authors           = CASE WHEN array_length(b.authors, 1) IS NULL
                                     THEN v.authors           ELSE b.authors           END,
            synopsis          = CASE WHEN b.synopsis          IS NULL
                                     THEN v.synopsis          ELSE b.synopsis          END,
            short_description = CASE WHEN b.short_description IS NULL
                                     THEN v.excerpt           ELSE b.short_description END,
            cover_image_url   = CASE WHEN b.cover_image_url   IS NULL
                                     THEN v.cover_url         ELSE b.cover_image_url   END,
            edition           = CASE WHEN b.edition           IS NULL
                                     THEN v.edition           ELSE b.edition           END,
            publish_date      = CASE WHEN b.publish_date      IS NULL
                                     THEN v.pub_date          ELSE b.publish_date      END,
            dewey_decimal     = CASE WHEN b.dewey_decimal      IS NULL
                                     THEN v.dewey             ELSE b.dewey_decimal     END,

            -- Overwrite: ISBNdb is the authoritative catalog source for these
            isbn10            = COALESCE(v.isbn10,    b.isbn10),
            publisher         = COALESCE(v.publisher, b.publisher),
            language          = COALESCE(v.language,  b.language),
            pages             = COALESCE(v.pages,     b.pages),

            -- Merge subjects: union of existing + ISBNdb, deduplicated
            subjects          = CASE
                                  WHEN v.subjects IS NULL THEN b.subjects
                                  WHEN array_length(b.subjects, 1) IS NULL THEN v.subjects
                                  ELSE (
                                    SELECT array_agg(DISTINCT s)
                                    FROM unnest(b.subjects || v.subjects) AS s
                                  )
                                END,

            cleaning_flags    = v.new_flags

        FROM (VALUES %s) AS v(book_id, authors, synopsis, excerpt, cover_url,
                               publisher, pages, language, isbn10, edition,
                               pub_date, subjects, dewey, new_flags)
        WHERE b.id = v.book_id
    """
    template = (
        "(%s::integer, %s::text[], %s::text, %s::text, %s::text, "
        "%s::text, %s::integer, %s::varchar, %s::varchar, %s::text, "
        "%s::date, %s::text[], %s::text[], %s::text[])"
    )
    with conn.cursor() as cur:
        execute_values(cur, sql, updates, template=template, page_size=DB_FLUSH_SIZE)
    conn.commit()


# ---------------------------------------------------------------------------
# Main enrichment loop
# ---------------------------------------------------------------------------

def enrich(conn, args) -> None:  # noqa: C901
    if not ISBNDB_API_KEY:
        print("ERROR: ISBNDB_API_KEY not set in .env", file=sys.stderr)
        sys.exit(1)

    rate_limiter = RateLimiter(daily_buffer=args.daily_buffer)

    if args.dry_run:
        print("Pre-flight quota check... skipped (dry-run mode, no real API calls).")
    else:
        print("Pre-flight quota check...")
        quota_left = preflight_quota_check()
        if quota_left == 0:
            print("No daily quota remaining. Try again after 00:00 UTC.")
            return
        if quota_left != -1 and quota_left <= args.daily_buffer:
            print(f"Only {quota_left} calls remaining (<= buffer {args.daily_buffer}). Stopping.")
            return
        if quota_left > 0:
            rate_limiter.daily_remaining = quota_left

    print("Fetching books to enrich from DB...")
    all_books = fetch_books_to_enrich(
        conn, args.limit, args.priority_only,
        embed_prep=args.embed_prep,
        review_boost=args.review_boost,
        max_reviews=args.max_reviews,
    )
    total = len(all_books)
    print(f"  {total:,} books queued for enrichment.")

    if not all_books:
        print("Nothing to do - all eligible books already processed.")
        return

    processable = min(total, max(0, rate_limiter.daily_remaining - args.daily_buffer))
    print(f"  Can process today (within quota): ~{processable:,}")
    if processable == 0:
        print("Quota too low to proceed safely. Try again after 00:00 UTC.")
        return
    print()

    # Counters
    enriched = 0
    not_found = 0
    checked_no_data = 0
    editorial_reviews_found = 0
    daily_stopped = False

    pending_book_updates: list[tuple] = []
    pending_editorial_reviews: list[ReviewRow] = []

    batches = [all_books[i:i + BULK_BATCH_SIZE] for i in range(0, len(all_books), BULK_BATCH_SIZE)]

    with tqdm(total=total, desc="ISBNdb enrichment", unit="book") as pbar:
        for batch in batches:
            if rate_limiter.daily_exhausted(batch_size=len(batch)):
                tqdm.write(
                    f"\n  [quota] {rate_limiter.daily_remaining} remaining "
                    f"(<= buffer {args.daily_buffer} + batch {len(batch)}). Stopping."
                )
                daily_stopped = True
                break

            isbn_list = [row[1] for row in batch]

            # In dry-run mode skip real API calls entirely - just simulate hits.
            if args.dry_run:
                found_map = {isbn: {"authors": ["Dry Run Author"],
                                    "synopsis": "Dry run placeholder synopsis."
                                    } for isbn in isbn_list}
                rate_limiter.daily_remaining = max(
                    0, rate_limiter.daily_remaining - len(batch)
                )
                pbar.update(len(batch))
                enriched += len(batch)
                pbar.set_postfix(enriched=enriched, not_found=not_found,
                                 editorial=editorial_reviews_found,
                                 quota=rate_limiter.daily_remaining)
                continue

            found_map = lookup_isbns_bulk(isbn_list, rate_limiter)

            # Pre-deduct so daily_exhausted() stays accurate next iteration.
            rate_limiter.daily_remaining = max(0, rate_limiter.daily_remaining - len(batch))

            for book_id, isbn13, old_flags in batch:
                bd = found_map.get(isbn13) or {}
                was_not_found = not bool(bd)

                was_enriched = False
                if was_not_found:
                    not_found += 1
                else:
                    authors = bd.get("authors") or []
                    synopsis = (bd.get("synopsis") or "").strip()
                    got_useful = bool(
                        authors or (synopsis and len(synopsis) >= MIN_DESCRIPTION_LEN)
                    )
                    if got_useful:
                        enriched += 1
                        was_enriched = True
                    else:
                        checked_no_data += 1

                    # Collect editorial reviews - store them separately.
                    editorial = extract_editorial_reviews(bd, isbn13)
                    if editorial:
                        editorial_reviews_found += len(editorial)
                        pending_editorial_reviews.extend(editorial)

                new_flags = compute_new_flags(old_flags, bd or None, was_not_found, was_enriched)

                # Parse publish_date from ISBNdb's YYYY, YYYY-MM, or YYYY-MM-DD formats.
                pub_date = None
                date_raw = bd.get("date_published") or ""
                if date_raw:
                    try:
                        parts = str(date_raw).strip().split("-")
                        year = int(parts[0]) if parts else None
                        month = int(parts[1]) if len(parts) > 1 else 1
                        day = int(parts[2]) if len(parts) > 2 else 1
                        if year and 1000 <= year <= 2100:
                            pub_date = date(year, max(1, min(month, 12)), max(1, min(day, 28)))
                    except (ValueError, IndexError):
                        pass

                pages_raw = bd.get("pages")
                try:
                    pages = int(pages_raw) if pages_raw else None
                except (ValueError, TypeError):
                    pages = None

                pending_book_updates.append((
                    book_id,
                    bd.get("authors") or None,
                    (bd.get("synopsis") or "").strip() or None,
                    (bd.get("excerpt") or "").strip() or None,   # -> short_description
                    bd.get("image") or None,                      # -> cover_image_url
                    bd.get("publisher") or None,                  # OVERWRITE
                    pages,                                         # OVERWRITE
                    bd.get("language") or None,                   # OVERWRITE
                    bd.get("isbn10") or None,                     # OVERWRITE
                    bd.get("edition") or None,
                    pub_date,
                    bd.get("subjects") or None,                   # MERGE
                    bd.get("dewey_decimal") or None,
                    new_flags,
                ))

            pbar.update(len(batch))
            pbar.set_postfix(
                enriched=enriched,
                not_found=not_found,
                editorial=editorial_reviews_found,
                quota=rate_limiter.daily_remaining,
            )

            # Flush book updates when buffer is full.
            if len(pending_book_updates) >= DB_FLUSH_SIZE:
                flush_book_updates(conn, pending_book_updates, args.dry_run)
                pending_book_updates.clear()

            # Flush editorial reviews when buffer is full.
            if len(pending_editorial_reviews) >= DB_FLUSH_SIZE:
                if not args.dry_run:
                    bulk_insert_reviews(conn, pending_editorial_reviews)
                pending_editorial_reviews.clear()

    # Final flush.
    if pending_book_updates:
        flush_book_updates(conn, pending_book_updates, args.dry_run)
    if pending_editorial_reviews and not args.dry_run:
        bulk_insert_reviews(conn, pending_editorial_reviews)

    print(f"\n{'='*55}")
    print(f"Enrichment run complete.")
    print(f"  Books enriched (authors or synopsis):  {enriched:,}")
    print(f"  Not found in ISBNdb:                   {not_found:,}")
    print(f"  Found but no new useful data:          {checked_no_data:,}")
    print(f"  Editorial reviews stored:              {editorial_reviews_found:,}")
    print(f"  Daily quota remaining:                 {rate_limiter.daily_remaining:,}")
    if daily_stopped:
        print(f"  Stopped early: quota buffer reached.")
        print(f"  Re-run tomorrow after 00:00 UTC to continue.")
    if args.dry_run:
        print("  (dry run - nothing written to DB)")
    print()
    print("Useful queries:")
    print("  -- Books successfully enriched:")
    print("  SELECT COUNT(*) FROM books WHERE cleaning_flags @> ARRAY['CLEANED'];")
    print()
    print("  -- Editorial reviews:")
    print("  SELECT COUNT(*) FROM reviews WHERE review_type = 'editorial';")
    print()
    print("  -- Books still needing enrichment:")
    print("  SELECT COUNT(*) FROM books")
    print("    WHERE cleaning_flags @> ARRAY['missing_description']")
    print("    AND NOT (cleaning_flags @> ARRAY['isbndb_checked'])")
    print("    AND NOT (cleaning_flags @> ARRAY['isbndb_not_found']);")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--limit", type=int, default=None,
                        help="Process at most N books (for testing).")
    parser.add_argument("--priority-only", action="store_true",
                        help="Only process missing_description books.")
    parser.add_argument("--embed-prep", action="store_true",
                        help="Target unembedded ucsd_graph books with 10+ reviews.")
    parser.add_argument("--review-boost", action="store_true",
                        help="Target unembedded goodreads_bbe/cmu books below --max-reviews, "
                             "ordered by review count DESC. ISBNdb adds editorial reviews.")
    parser.add_argument("--max-reviews", type=int, default=60,
                        help="Upper review cap for --review-boost mode (default 60).")
    parser.add_argument("--daily-buffer", type=int, default=DEFAULT_DAILY_BUFFER,
                        help=f"Stop when daily remaining <= this (default {DEFAULT_DAILY_BUFFER}).")
    parser.add_argument("--dry-run", action="store_true",
                        help="Make API calls but write nothing to the DB.")
    args = parser.parse_args()

    conn = get_connection()
    try:
        enrich(conn, args)
    finally:
        conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
