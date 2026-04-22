"""
Ingest the UCSD Goodreads Book Graph dataset into the books and reviews tables.

This is the third and final source ingestor. It handles two tasks:
    1. BOOK INGEST: Stream the UCSD books file, apply quality filters, and
       either UPDATE existing rows (fill NULL fields only) or INSERT new rows
       with cleaning_flags marking any fixable quality issues.
    2. REVIEW INGESTION: Stream the UCSD reviews file and insert user reviews
       keyed on isbn13.

REQUIRED FILES (download with: uv run python scripts/download_datasets.py --ucsd):
    data/raw/ucsd_goodreads_books.json.gz    (~250 MB compressed)
    data/raw/ucsd_goodreads_reviews.json.gz  (~large, potentially several GB)

FORMAT QUIRKS (this is the messy dataset mentioned in the paper):
    - Both files are gzipped newline-delimited JSON (.jsonl inside .gz).
      Each line is one self-contained JSON object. Do NOT load the whole
      file into memory; stream it line-by-line.
    - The books file has:
        - `isbn` (10-digit ISBN, often missing or "")
        - `isbn13` (13-digit ISBN, often missing)
        - `title` (may contain HTML artifacts like &amp; or &#39;)
        - `authors` (list of dicts with `author_id` and `name`)
        - `popular_shelves` (list of dicts with `name` and `count` - these
           are the genre tags, NOT a clean categorical list)
        - `description` (may contain raw HTML tags)
        - `publication_year`, `publication_month`, `publication_day` (ints)
        - `publisher` (string)
        - `num_pages` (string, may be "")
        - `average_rating`, `ratings_count` (strings)
    - The reviews file has:
        - `book_id` (UCSD internal ID, not ISBN - requires join against books file)
        - `user_id` (string)
        - `rating` (int, 0-5)
        - `review_text` (string, may be empty)
        - `date_added`, `date_updated` (string timestamps)
        - `spoiler` (bool)

QUALITY FILTERS applied during book ingest:
    Hard rejections (row is skipped entirely):
        - num_ratings < MIN_RATINGS (10) - too sparse for useful recommendations
        - num_ratings is None - can't verify quality
        - language_code is a known non-English value (spa, ita, ara, ger, ...) -
          the embedding model is BGE-base-en-v1.5 (English only); non-English text
          produces meaningless vectors that pollute recommendation results
        - Empty title (caught by ucsd_book_to_row returning None)

    Soft flags (row is inserted but marked for later cleaning):
        - "missing_isbn"        : no ISBN10 or ISBN13 anywhere in the record
        - "missing_description" : no synopsis/description field
        - "short_description"   : description exists but < MIN_DESCRIPTION_LEN (50) chars
        - "missing_author"      : no author attribution (dedup key is title-only)
        - "suspect_title"       : title < 3 chars or purely numeric
        - "low_rating_count"    : 10 <= num_ratings < 25 (borderline quality)

    Note on language: books with absent language_code are accepted (overwhelmingly
    English Goodreads entries). Only explicitly non-English codes trigger rejection.

    Flagged rows can later be enriched via ISBNdb API or deleted if unfixable.
    Query them with: SELECT * FROM books WHERE cleaning_flags IS NOT NULL

DEDUPLICATION STRATEGY:
    Pass 1 builds a match_index from existing DB rows. For each UCSD book:
        1. ISBN13 exact match -> batch UPDATE the existing row (NULL fields only)
        2. (title, author) fuzzy match -> same
        3. (title, author) matches a no-isbn DB book -> skip (no key to update by)
        4. No match -> INSERT as a new row with cleaning_flags set

    Within-run duplicates are prevented by adding inserted books back to the
    match_index immediately, so the same title/ISBN won't be inserted twice
    even if the UCSD file contains it multiple times.

    UPDATE semantics: we never overwrite existing data. Each field uses:
        CASE WHEN existing IS NULL THEN ucsd_value ELSE existing END

    Batch UPDATEs use a single UPDATE ... FROM VALUES round trip per
    BULK_PAGE_SIZE rows instead of N individual UPDATE statements. This is
    critical for the 2.3M row full dataset where matches can be in the tens
    of thousands.

REVIEW PASS:
    Reviews reference books by UCSD `book_id`, not ISBN. We build a
    book_id -> isbn13 map during the books pass, then use it in the reviews
    pass. Only reviews for books that resolved to a known isbn13 are inserted.
    Reviews with empty text are skipped (no value to the embedding pipeline).

Usage:
    uv run python scripts/ingest_ucsd_graph.py
    uv run python scripts/ingest_ucsd_graph.py --books-only
    uv run python scripts/ingest_ucsd_graph.py --reviews-only
    uv run python scripts/ingest_ucsd_graph.py --enrich-only   # update existing, skip new inserts
    uv run python scripts/ingest_ucsd_graph.py --max-reviews 10
    uv run python scripts/ingest_ucsd_graph.py --dry-run
    uv run python scripts/ingest_ucsd_graph.py --limit 10000   # test subset
"""

from __future__ import annotations

import argparse
import gzip
import html
import json
import re
import sys
from collections import defaultdict
from datetime import date, datetime
from pathlib import Path
from typing import Optional

import psycopg2
from psycopg2.extras import execute_values
from tqdm import tqdm

import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from lib.db import (
    BULK_PAGE_SIZE,
    bulk_insert_books,
    bulk_insert_reviews,
    ensure_cleaning_flags_column,
    ensure_ucsd_book_id_column,
    ensure_review_book_id_column,
    get_connection,
)
from lib.isbn import normalize_isbn13
from lib.models import BookRow, BookSource, ReviewRow, ReviewSource


BOOKS_FILE = Path("data/raw/ucsd_goodreads_books.json.gz")
REVIEWS_FILE = Path("data/raw/ucsd_goodreads_reviews.json.gz")

# Hard reject threshold: books with fewer ratings are too sparse to be useful.
MIN_RATINGS = 10

# Soft flag threshold: descriptions shorter than this get flagged as "short_description".
MIN_DESCRIPTION_LEN = 50

# Shelf names that look like genres (not personal library-management names).
_GENRE_SHELF_RE = re.compile(
    r"^(?!to-read|currently-reading|owned|favorites|books-i-own|"
    r"re-read|on-hold|maybe|wish-list|library|read|dnf|abandoned)",
    re.IGNORECASE,
)
# Minimum shelf count to be treated as a real genre tag.
_MIN_SHELF_COUNT = 5

_HTML_TAG_RE = re.compile(r"<[^>]+>")


def clean_html(text: str) -> str:
    """Strip HTML tags and unescape entities."""
    text = _HTML_TAG_RE.sub(" ", text)
    text = html.unescape(text)
    return re.sub(r"\s+", " ", text).strip()


def extract_genres(popular_shelves: list[dict]) -> list[str]:
    """Filter popular_shelves to only shelf names that look like genre tags."""
    genres = []
    seen: set[str] = set()
    for shelf in popular_shelves:
        name = shelf.get("name", "").strip()
        count_raw = shelf.get("count", 0)
        try:
            count = int(count_raw)
        except (ValueError, TypeError):
            count = 0
        if not name or count < _MIN_SHELF_COUNT:
            continue
        if _GENRE_SHELF_RE.match(name) and name not in seen:
            seen.add(name)
            genres.append(name)
    # Cap at 20 to avoid runaway arrays.
    return genres[:20]


def extract_authors(authors_list: list[dict]) -> list[str]:
    """Extract author name strings from the UCSD nested author dicts."""
    return [
        str(a.get("name", "")).strip()
        for a in authors_list
        if a.get("name")
    ]


def parse_ucsd_date(year_raw, month_raw, day_raw) -> Optional[date]:
    """Build a date from separate year/month/day fields, all potentially empty."""
    try:
        year = int(year_raw) if year_raw else None
        month = int(month_raw) if month_raw else 1
        day = int(day_raw) if day_raw else 1
        if year and 1000 <= year <= 2100:
            return date(year, max(1, min(month, 12)), max(1, min(day, 28)))
    except (ValueError, TypeError):
        pass
    return None


def ucsd_book_to_row(obj: dict) -> Optional[tuple[BookRow, str]]:
    """Convert one UCSD books JSONL object to a BookRow.

    Returns None only if there is no usable title (hard skip).
    Quality flagging happens separately in compute_cleaning_flags().

    NOTE ON AUTHORS: The full UCSD dataset is privacy-scrubbed - the authors
    field only contains {author_id, role} dicts, never a name. extract_authors()
    returns [] for all books from this file. The missing_author flag will be
    set on every new UCSD insert; ISBNdb enrichment (using isbn13) is the
    path to fill author names later.
    """
    title = clean_html(obj.get("title", "").strip())
    if not title:
        return None

    isbn13 = normalize_isbn13(obj.get("isbn13")) or normalize_isbn13(obj.get("isbn"))
    authors = extract_authors(obj.get("authors", []))
    genres = extract_genres(obj.get("popular_shelves", []))

    desc_raw = obj.get("description", "")
    synopsis = clean_html(desc_raw) if desc_raw else None

    rating_raw = obj.get("average_rating", "")
    try:
        rating = float(rating_raw) if rating_raw else None
        rating = rating if rating is not None and 0 <= rating <= 5 else None
    except ValueError:
        rating = None

    num_ratings_raw = obj.get("ratings_count", "")
    try:
        num_ratings = int(num_ratings_raw) if num_ratings_raw else None
    except ValueError:
        num_ratings = None

    pages_raw = obj.get("num_pages", "")
    try:
        pages = int(pages_raw) if pages_raw else None
        pages = pages if pages and pages > 0 else None
    except ValueError:
        pages = None

    pub_date = parse_ucsd_date(
        obj.get("publication_year"),
        obj.get("publication_month"),
        obj.get("publication_day"),
    )

    # UCSD uses ISO 639-2 codes (eng, spa, ita) and IETF tags (en-US, en-GB).
    # None means absent from the record - treated as likely English downstream.
    language = obj.get("language_code") or None

    # Cover image URL - free data, useful for recommendation UI.
    cover_url = obj.get("image_url") or None
    # UCSD image URLs sometimes point to placeholder "nophoto" images.
    if cover_url and "nophoto" in cover_url:
        cover_url = None

    # title_without_series is cleaner for dedup matching (strips " (#3)" etc.)
    # We store the full title but expose the clean one for match key computation.
    title_clean = clean_html(obj.get("title_without_series", "").strip()) or title

    row = BookRow(
        isbn13=isbn13,
        isbn10=None,
        title=title,
        authors=authors,
        publisher=clean_html(obj.get("publisher", "")) or None,
        publish_date=pub_date,
        genres=genres,
        subjects=[],
        language=language,
        synopsis=synopsis,
        pages=pages,
        goodreads_rating=rating,
        num_ratings=num_ratings,
        cover_image_url=cover_url,
        source=BookSource.UCSD_GRAPH,
    )
    # Return the series-stripped title alongside the row. It's cleaner for dedup
    # matching (e.g. "Harry Potter and the Chamber of Secrets" instead of
    # "Harry Potter and the Chamber of Secrets (Harry Potter #2)").
    return row, title_clean


_ENGLISH_LANG_CODES = frozenset(
    ("en", "eng", "en-us", "en-gb", "en-ca", "en-au", "english")
)

_NON_ENGLISH_HARD_REJECT = re.compile(
    r"^(?!en\b|eng$)(spa|ita|ara|ger|fre|por|ind|tur|per|fin|swe|gre|cze|"
    r"jpn|kor|chi|zho|rus|pol|hun|dan|nor|heb|ukr|cat|ron|slk|hrv|lit|lav|"
    r"est|vie|tha|ben|hin|tam|tel|mal|mar|urd|sin|bul|srp|mkd|sqi|isl|gle|"
    r"lat|eus|glg|afr|swa|nld|nl|gle|msa|fil|tgl)",
    re.IGNORECASE,
)


def is_non_english(language: str | None) -> bool:
    """Return True if language is explicitly set to a non-English value.

    Absent language (None or "") is treated as likely English - the UCSD
    dataset is heavily English-biased and missing language_code is common.
    We only reject when the language is *explicitly* set to non-English.
    """
    if not language:
        return False
    lang = language.lower().strip()
    if lang in _ENGLISH_LANG_CODES:
        return False
    return bool(_NON_ENGLISH_HARD_REJECT.match(lang))


def compute_cleaning_flags(row: BookRow) -> list[str]:
    """Return quality flags for a UCSD book row that passed hard rejections.

    These flags mark fixable issues. An empty list means the row is clean.
    ISBNdb enrichment or manual review can resolve the flagged fields later.

    Non-English books are hard-rejected before this is called (they never
    reach flag computation), so no language flag appears here.
    """
    flags: list[str] = []

    if not row.isbn13:
        # No ISBN - can't do catalog lookups or ISBNdb enrichment.
        flags.append("missing_isbn")

    if not row.synopsis:
        # No text at all - embedding will be title+genres only (weak signal).
        flags.append("missing_description")
    elif len(row.synopsis) < MIN_DESCRIPTION_LEN:
        # Very short description - embedding quality will be limited.
        flags.append("short_description")

    if not row.authors:
        # No author attribution - dedup is title-only, higher collision risk.
        flags.append("missing_author")

    # Suspicious title (very short or purely numeric).
    if len(row.title.strip()) < 3 or row.title.strip().isdigit():
        flags.append("suspect_title")

    # Borderline rating count - passed the hard floor but still thin.
    if row.num_ratings is not None and MIN_RATINGS <= row.num_ratings < 25:
        flags.append("low_rating_count")

    return flags


def ucsd_review_to_row(
    obj: dict,
    isbn13: Optional[str] = None,
    book_id: Optional[int] = None,
) -> Optional[ReviewRow]:
    """Convert one UCSD reviews JSONL object to a ReviewRow."""
    review_text = obj.get("review_text", "").strip()
    # Skip empty reviews - no value to the embedding pipeline.
    if not review_text:
        return None

    rating_raw = obj.get("rating")
    try:
        rating = int(rating_raw) if rating_raw is not None else None
        # DB constraint: rating >= 1. UCSD uses 0 for "no rating" - treat as NULL.
        rating = rating if rating is not None and 1 <= rating <= 5 else None
    except (ValueError, TypeError):
        rating = None

    date_posted: Optional[date] = None
    for date_field in ("date_updated", "date_added"):
        raw = obj.get(date_field, "")
        if raw:
            try:
                date_posted = datetime.strptime(raw, "%a %b %d %H:%M:%S %z %Y").date()
                break
            except (ValueError, TypeError):
                pass

    return ReviewRow(
        isbn13=isbn13,
        book_id=book_id,
        user_id=str(obj.get("user_id", "")).strip() or None,
        rating=rating,
        review_text=review_text,
        date_posted=date_posted,
        spoiler_flag=bool(obj.get("spoiler", False)),
        source=ReviewSource.UCSD_GRAPH,
    )


def normalize_for_match(text: str) -> str:
    """Normalize a title or author name for fuzzy deduplication matching."""
    _STRIP_PUNCT = re.compile(r"[^\w\s]")
    _ARTICLES = re.compile(r"^(the|a|an)\s+", re.IGNORECASE)
    text = text.lower().strip()
    text = _ARTICLES.sub("", text)
    text = html.unescape(text)
    text = _STRIP_PUNCT.sub("", text)
    return re.sub(r"\s+", " ", text).strip()


def build_match_index(conn) -> dict[str, tuple[Optional[str], Optional[int]]]:
    """Build a lookup index from all existing books rows.

    Returns a dict where each value is (isbn13_or_None, db_id_or_None):
        isbn13 -> (isbn13, db_id)          for books that have an isbn13
        title_author_key -> (isbn13, db_id) same books, title+author fallback
        title_author_key -> (None, db_id)   for no-isbn books

    db_id is books.id and lets us link reviews for no-isbn books via
    reviews.book_id. Previously these books had no review FK path at all.

    Callers MUST use `key in index` to check existence, NOT `.get(key)`.
    """
    index: dict[str, tuple[Optional[str], Optional[int]]] = {}
    with conn.cursor() as cur:
        cur.execute(
            "SELECT id, isbn13, title, authors FROM books WHERE isbn13 IS NOT NULL"
        )
        for db_id, isbn13, title, authors in cur.fetchall():
            index[isbn13] = (isbn13, db_id)
            first_author = (authors[0] if authors else "") if authors else ""
            key = normalize_for_match(title or "") + "|" + normalize_for_match(first_author)
            if key not in index:
                index[key] = (isbn13, db_id)

        # No-isbn books: store (None, db_id) so callers can still get db_id
        # and update ucsd_book_id / link reviews via book_id.
        cur.execute(
            "SELECT id, title, authors FROM books WHERE isbn13 IS NULL"
        )
        for db_id, title, authors in cur.fetchall():
            first_author = (authors[0] if authors else "") if authors else ""
            key = normalize_for_match(title or "") + "|" + normalize_for_match(first_author)
            if key not in index:
                index[key] = (None, db_id)

    return index


def _flush_book_updates(conn, updates: list[tuple]) -> None:
    """Batch-update existing book rows, filling NULL fields only.

    Each tuple in `updates`:
        (isbn13, genres, synopsis, goodreads_rating, num_ratings,
         publisher, publish_date, pages, ucsd_book_id)

    Uses a single UPDATE ... FROM VALUES round trip instead of one UPDATE per
    matched row. Without this, matching 50K+ existing rows would mean 50K+
    Neon round trips (~hours). With this, it's one round trip per batch.

    Fields are only written when the existing DB value is NULL. We never
    overwrite data from a higher-quality source. ucsd_book_id is stored so
    we can link reviews for no-isbn books in subsequent runs.
    """
    if not updates:
        return

    sql = """
        UPDATE books AS b SET
            genres           = CASE WHEN array_length(b.genres, 1) IS NULL
                                    THEN v.genres           ELSE b.genres           END,
            synopsis         = CASE WHEN b.synopsis         IS NULL
                                    THEN v.synopsis         ELSE b.synopsis         END,
            goodreads_rating = CASE WHEN b.goodreads_rating IS NULL
                                    THEN v.goodreads_rating ELSE b.goodreads_rating END,
            num_ratings      = CASE WHEN b.num_ratings      IS NULL
                                    THEN v.num_ratings      ELSE b.num_ratings      END,
            publisher        = CASE WHEN b.publisher        IS NULL
                                    THEN v.publisher        ELSE b.publisher        END,
            publish_date     = CASE WHEN b.publish_date     IS NULL
                                    THEN v.pub_date         ELSE b.publish_date     END,
            pages            = CASE WHEN b.pages            IS NULL
                                    THEN v.pages            ELSE b.pages            END,
            ucsd_book_id     = COALESCE(b.ucsd_book_id, v.ucsd_book_id)
        FROM (VALUES %s) AS v(isbn13, genres, synopsis, goodreads_rating,
                               num_ratings, publisher, pub_date, pages, ucsd_book_id)
        WHERE b.isbn13 = v.isbn13
    """
    template = (
        "(%s::varchar(13), %s::text[], %s::text, "
        "%s::double precision, %s::integer, %s::text, %s::date, %s::integer, %s::text)"
    )
    with conn.cursor() as cur:
        execute_values(cur, sql, updates, template=template, page_size=BULK_PAGE_SIZE)
    conn.commit()


def _flush_no_isbn_ucsd_id_updates(conn, updates: list[tuple]) -> None:
    """Set ucsd_book_id on matched no-isbn books.

    Each tuple: (db_id, ucsd_book_id)

    No-isbn books are matched by title+author but have no isbn13 to UPDATE by,
    so they get a separate batch UPDATE keyed on books.id. We store ucsd_book_id
    so the reviews pass can link their reviews via reviews.book_id.
    """
    if not updates:
        return

    sql = """
        UPDATE books AS b
        SET ucsd_book_id = COALESCE(b.ucsd_book_id, v.ucsd_id)
        FROM (VALUES %s) AS v(db_id, ucsd_id)
        WHERE b.id = v.db_id::integer
    """
    template = "(%s::integer, %s::text)"
    with conn.cursor() as cur:
        execute_values(cur, sql, updates, template=template, page_size=BULK_PAGE_SIZE)
    conn.commit()


def build_ucsd_to_db_id_map(conn) -> dict[str, int]:
    """Query ucsd_book_id -> books.id for all rows that have ucsd_book_id set.

    Called after the books pass completes. Covers:
    - Newly inserted no-isbn books (ucsd_book_id set at insert time)
    - Matched no-isbn books (ucsd_book_id set by _flush_no_isbn_ucsd_id_updates)
    - Matched isbn books (ucsd_book_id set by _flush_book_updates)

    The result is used in the reviews pass to route no-isbn book reviews
    through reviews.book_id instead of reviews.isbn13.
    """
    with conn.cursor() as cur:
        cur.execute(
            "SELECT ucsd_book_id, id FROM books WHERE ucsd_book_id IS NOT NULL"
        )
        return {ucsd_id: db_id for ucsd_id, db_id in cur.fetchall()}


def ingest_books(
    conn,
    match_index: dict[str, tuple[Optional[str], Optional[int]]],
    book_id_to_isbn: dict[str, str],
    book_id_to_db_id: dict[str, int],
    limit: Optional[int],
    dry_run: bool,
    enrich_only: bool = False,
) -> None:
    """Stream the UCSD books file, apply quality filters, update or insert rows.

    Populates book_id_to_isbn in-place for the reviews pass. Only books with
    a resolved isbn13 are added to this map (no-isbn books can't have reviews
    linked to them via FK).

    See module docstring for full quality filter and deduplication details.
    """
    print(f"\nPass 1: streaming {BOOKS_FILE.name}...")
    if enrich_only:
        print("  (enrich-only mode: updating existing rows, skipping new inserts)")

    matched = 0               # Existing rows updated
    matched_no_isbn_db = 0    # Existing no-isbn rows - skipped (no key to update by)
    new_inserts = 0           # New rows inserted
    flagged_inserts = 0       # New inserts that carry at least one cleaning flag
    skipped_no_title = 0      # Rejected: empty title or JSON parse error
    skipped_low_ratings = 0   # Rejected: num_ratings < MIN_RATINGS or None
    skipped_non_english = 0   # Rejected: explicit non-English language code
    skipped_new = 0           # enrich_only mode: new books skipped
    flag_tally: dict[str, int] = {}  # Per-flag counts for the summary

    pending_inserts: list[BookRow] = []
    # Each update tuple: (isbn13, genres, synopsis, rating, num_ratings,
    #                     publisher, publish_date, pages, ucsd_book_id)
    pending_updates: list[tuple] = []
    # For no-isbn books matched by title+author: (db_id, ucsd_book_id)
    pending_no_isbn_ucsd_updates: list[tuple] = []

    total = 0
    with gzip.open(BOOKS_FILE, "rt", encoding="utf-8", errors="replace") as f:
        for line in tqdm(f, desc="UCSD books", unit="row"):
            if limit and total >= limit:
                break
            line = line.strip()
            if not line:
                continue

            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                skipped_no_title += 1
                continue

            total += 1
            book_id = str(obj.get("book_id", ""))

            parsed = ucsd_book_to_row(obj)
            if parsed is None:
                skipped_no_title += 1
                continue
            row, title_clean = parsed

            # Hard reject: too few ratings, or unknown count (can't verify quality).
            if row.num_ratings is None or row.num_ratings < MIN_RATINGS:
                skipped_low_ratings += 1
                continue

            # Hard reject: explicitly non-English language. BGE-base-en-v1.5
            # produces garbage embeddings for non-English text - don't pollute
            # the recommendation index.
            if is_non_english(row.language):
                skipped_non_english += 1
                continue

            # Compute (title, author) key used for both matching and dedup.
            first_author = row.authors[0] if row.authors else ""
            ta_key = normalize_for_match(title_clean) + "|" + normalize_for_match(first_author)

            # --- Check if this book already exists in the DB ---
            resolved_isbn: Optional[str] = None
            resolved_db_id: Optional[int] = None
            book_exists = False

            if row.isbn13 and row.isbn13 in match_index:
                resolved_isbn, resolved_db_id = match_index[row.isbn13]
                book_exists = True
            elif ta_key in match_index:
                resolved_isbn, resolved_db_id = match_index[ta_key]
                book_exists = True

            if book_exists:
                if resolved_isbn:
                    # Matched an existing isbn row. Queue a batch UPDATE to
                    # fill NULL fields and store ucsd_book_id for review linking.
                    book_id_to_isbn[book_id] = resolved_isbn
                    if resolved_db_id:
                        book_id_to_db_id[book_id] = resolved_db_id
                    matched += 1
                    if not dry_run:
                        pending_updates.append((
                            resolved_isbn,
                            row.genres or None,
                            row.synopsis,
                            row.goodreads_rating,
                            row.num_ratings,
                            row.publisher,
                            row.publish_date,
                            row.pages,
                            book_id,   # ucsd_book_id
                        ))
                        if len(pending_updates) >= BULK_PAGE_SIZE:
                            _flush_book_updates(conn, pending_updates)
                            pending_updates.clear()
                else:
                    # Matched an existing no-isbn book. Store ucsd_book_id so
                    # the reviews pass can link reviews via book_id.
                    matched_no_isbn_db += 1
                    if resolved_db_id:
                        book_id_to_db_id[book_id] = resolved_db_id
                        if not dry_run:
                            pending_no_isbn_ucsd_updates.append((resolved_db_id, book_id))
                            if len(pending_no_isbn_ucsd_updates) >= BULK_PAGE_SIZE:
                                _flush_no_isbn_ucsd_id_updates(conn, pending_no_isbn_ucsd_updates)
                                pending_no_isbn_ucsd_updates.clear()
                continue

            # --- New book not in DB ---
            if enrich_only:
                # enrich_only mode only touches existing rows. Don't add to
                # book_id_to_isbn either - reviews would FK-violate.
                skipped_new += 1
                continue

            # Compute soft quality flags for this new row.
            flags = compute_cleaning_flags(row)

            # Store ucsd_book_id so we can look up books.id after the bulk
            # insert and link reviews for no-isbn books.
            row.ucsd_book_id = book_id

            # Add to match_index immediately so a duplicate later in the same
            # file doesn't get inserted a second time. db_id is None because
            # we don't have it until after the bulk insert; build_ucsd_to_db_id_map
            # fills the gap after the full books pass.
            match_index[ta_key] = (row.isbn13, None)
            if row.isbn13:
                match_index[row.isbn13] = (row.isbn13, None)
                book_id_to_isbn[book_id] = row.isbn13

            row.cleaning_flags = flags
            if flags:
                flagged_inserts += 1
                for f in flags:
                    flag_tally[f] = flag_tally.get(f, 0) + 1
            new_inserts += 1
            pending_inserts.append(row)

            if len(pending_inserts) >= BULK_PAGE_SIZE and not dry_run:
                bulk_insert_books(conn, pending_inserts, on_conflict="DO NOTHING")
                pending_inserts.clear()

    # Flush any remaining batches.
    if pending_updates and not dry_run:
        _flush_book_updates(conn, pending_updates)
    if pending_no_isbn_ucsd_updates and not dry_run:
        _flush_no_isbn_ucsd_id_updates(conn, pending_no_isbn_ucsd_updates)
    if pending_inserts and not dry_run:
        bulk_insert_books(conn, pending_inserts, on_conflict="DO NOTHING")

    total_rejected = skipped_low_ratings + skipped_non_english + skipped_no_title
    total_accepted = matched + matched_no_isbn_db + new_inserts
    print(f"  Matched (updated):              {matched:,}")
    print(f"  Matched (no-isbn, skipped):     {matched_no_isbn_db:,}")
    print(f"  New inserts:                    {new_inserts:,}")
    if flagged_inserts:
        print(f"    of which flagged:             {flagged_inserts:,}")
        for flag, count in sorted(flag_tally.items(), key=lambda x: -x[1]):
            print(f"      {flag:<25} {count:,}")
    print(f"  --- Rejected ({total_rejected:,} total) ---")
    print(f"  Rejected (low ratings < {MIN_RATINGS}):    {skipped_low_ratings:,}")
    print(f"  Rejected (non-English):         {skipped_non_english:,}")
    print(f"  Rejected (no title/JSON err):   {skipped_no_title:,}")
    if enrich_only:
        print(f"  Skipped new (enrich-only):      {skipped_new:,}")
    print(f"  book_id -> isbn13 map:          {len(book_id_to_isbn):,} entries")


def load_existing_review_counts(
    conn,
    isbn_set: set[str],
    db_id_set: set[int],
) -> dict[str, int]:
    """Query the DB for how many UCSD reviews already exist per book.

    Returns a dict keyed by "isbn:<isbn13>" or "id:<db_id>" so both FK paths
    share a single cap counter. Used to pre-populate before streaming so a
    re-run tops books up to the cap rather than duplicating existing reviews.
    Only counts source='ucsd_graph' rows.
    """
    counts: dict[str, int] = {}
    with conn.cursor() as cur:
        if isbn_set:
            cur.execute(
                """
                SELECT isbn13, COUNT(*)
                FROM reviews
                WHERE isbn13 = ANY(%s) AND source = 'ucsd_graph'
                GROUP BY isbn13
                """,
                (list(isbn_set),),
            )
            for isbn13, count in cur.fetchall():
                counts[f"isbn:{isbn13}"] = count

        if db_id_set:
            cur.execute(
                """
                SELECT book_id, COUNT(*)
                FROM reviews
                WHERE book_id = ANY(%s) AND source = 'ucsd_graph'
                GROUP BY book_id
                """,
                (list(db_id_set),),
            )
            for db_id, count in cur.fetchall():
                counts[f"id:{db_id}"] = count

    return counts


def ingest_reviews(
    conn,
    book_id_to_isbn: dict[str, str],
    book_id_to_db_id: dict[str, int],
    max_reviews_per_book: int,
    limit: Optional[int],
    dry_run: bool,
) -> None:
    """Stream the UCSD reviews file and insert into the reviews table.

    Routes each review through one of two FK paths:
    - isbn books: review.isbn13 set, cap keyed by "isbn:<isbn13>"
    - no-isbn books: review.book_id set, cap keyed by "id:<db_id>"

    Pre-loads existing counts so re-runs top books up to cap rather than
    duplicating reviews already stored.
    """
    print(f"\nPass 2: streaming {REVIEWS_FILE.name}...")

    isbn_set = set(book_id_to_isbn.values())
    db_id_set = set(book_id_to_db_id.values())

    if conn and (isbn_set or db_id_set):
        existing_counts = load_existing_review_counts(conn, isbn_set, db_id_set)
        already_stored = sum(existing_counts.values())
        at_cap = sum(1 for c in existing_counts.values() if c >= max_reviews_per_book)
        print(f"  Pre-loaded {already_stored:,} existing reviews across "
              f"{len(existing_counts):,} books ({at_cap:,} already at cap).")
        print(f"  isbn-path books: {len(isbn_set):,}  |  "
              f"book_id-path (no-isbn): {len(db_id_set):,}")
    else:
        existing_counts = {}

    reviews_per_book: dict[str, int] = defaultdict(int, existing_counts)
    inserted = 0
    skipped_no_match = 0
    skipped_empty = 0
    skipped_cap = 0

    pending: list[ReviewRow] = []
    total = 0

    with gzip.open(REVIEWS_FILE, "rt", encoding="utf-8", errors="replace") as f:
        for line in tqdm(f, desc="UCSD reviews", unit="row"):
            if limit and total >= limit:
                break
            line = line.strip()
            if not line:
                continue

            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                skipped_empty += 1
                continue

            total += 1
            ucsd_id = str(obj.get("book_id", ""))

            # Try isbn path first, then no-isbn path.
            isbn13 = book_id_to_isbn.get(ucsd_id)
            if isbn13:
                cap_key = f"isbn:{isbn13}"
                review_isbn = isbn13
                review_db_id = None
            else:
                db_id = book_id_to_db_id.get(ucsd_id)
                if not db_id:
                    skipped_no_match += 1
                    continue
                cap_key = f"id:{db_id}"
                review_isbn = None
                review_db_id = db_id

            if reviews_per_book[cap_key] >= max_reviews_per_book:
                skipped_cap += 1
                continue

            review = ucsd_review_to_row(obj, isbn13=review_isbn, book_id=review_db_id)
            if review is None:
                skipped_empty += 1
                continue

            reviews_per_book[cap_key] += 1
            inserted += 1
            pending.append(review)

            if len(pending) >= BULK_PAGE_SIZE and not dry_run:
                bulk_insert_reviews(conn, pending)
                pending.clear()

    if pending and not dry_run:
        bulk_insert_reviews(conn, pending)

    topped_up = sum(
        1 for key, count in reviews_per_book.items()
        if existing_counts.get(key, 0) > 0 and count > existing_counts.get(key, 0)
    )
    print(f"  Reviews inserted:        {inserted:,}")
    if topped_up:
        print(f"  Books topped up:         {topped_up:,} (had reviews, got more)")
    print(f"  Skipped (no book match): {skipped_no_match:,}")
    print(f"  Skipped (empty text):    {skipped_empty:,}")
    print(f"  Skipped (cap reached):   {skipped_cap:,}")
    print(f"  Unique books covered:    {len(reviews_per_book):,}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--books-only", action="store_true",
                        help="Run the books pass only, skip reviews.")
    parser.add_argument("--reviews-only", action="store_true",
                        help="Run the reviews pass only, skip book inserts/updates.")
    parser.add_argument(
        "--enrich-only", action="store_true",
        help="Books pass only updates existing rows (fills NULL fields). "
             "New UCSD-only books are not inserted. Use this for a lightweight "
             "enrichment pass without adding new rows.",
    )
    parser.add_argument(
        "--max-reviews", type=int, default=60,
        help="Max reviews to ingest per book (default 60). More reviews give "
             "the review_embedding a richer signal during aggregation.",
    )
    parser.add_argument("--dry-run", action="store_true",
                        help="Parse and count without writing anything to the DB.")
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Process at most N lines per file (for quick smoke tests).",
    )
    args = parser.parse_args()

    do_books = not args.reviews_only
    do_reviews = not args.books_only

    if do_books and not BOOKS_FILE.exists():
        print(
            f"ERROR: {BOOKS_FILE} not found.\n"
            "Run: uv run python scripts/download_datasets.py --ucsd",
            file=sys.stderr,
        )
        return 1
    if do_reviews and not REVIEWS_FILE.exists():
        print(
            f"ERROR: {REVIEWS_FILE} not found.\n"
            "Run: uv run python scripts/download_datasets.py --ucsd",
            file=sys.stderr,
        )
        return 1

    conn = None if args.dry_run else get_connection()

    # Ensure all required columns exist before any reads or writes.
    # All three are idempotent (ADD COLUMN IF NOT EXISTS).
    if conn:
        ensure_cleaning_flags_column(conn)
        ensure_ucsd_book_id_column(conn)
        ensure_review_book_id_column(conn)
        print("Schema: cleaning_flags, ucsd_book_id, reviews.book_id confirmed.")

    book_id_to_isbn: dict[str, str] = {}
    book_id_to_db_id: dict[str, int] = {}

    if do_books:
        if not args.dry_run:
            print("Building match index from existing books table...")
            match_index = build_match_index(conn)
            print(f"  {len(match_index):,} entries indexed.")
        else:
            match_index = {}
        ingest_books(
            conn, match_index, book_id_to_isbn, book_id_to_db_id,
            args.limit, args.dry_run,
            enrich_only=args.enrich_only,
        )

    if do_reviews:
        if not book_id_to_isbn and not book_id_to_db_id:
            # reviews-only mode: rebuild both maps by re-scanning the books file.
            if not BOOKS_FILE.exists():
                print(
                    "ERROR: --reviews-only requires the books file to build the maps.",
                    file=sys.stderr,
                )
                return 1
            print("Building book_id maps from books file...")
            match_index = {} if args.dry_run else build_match_index(conn)
            ingest_books(
                conn, match_index, book_id_to_isbn, book_id_to_db_id,
                limit=None, dry_run=True,
                enrich_only=True,
            )

        # After the books pass, query the DB for ucsd_book_id -> books.id to
        # cover newly inserted no-isbn books (their db_id was unknown at insert
        # time). Merge into book_id_to_db_id which already has matched rows.
        if conn:
            print("Building ucsd_book_id -> books.id map from DB...")
            ucsd_db_map = build_ucsd_to_db_id_map(conn)
            # Only add entries for ucsd_ids not already in the map (matched
            # rows were added during the books pass with correct db_ids).
            for ucsd_id, db_id in ucsd_db_map.items():
                if ucsd_id not in book_id_to_db_id:
                    book_id_to_db_id[ucsd_id] = db_id
            # Remove isbn books from db_id map - they use the isbn path.
            # This avoids double-counting in cap logic.
            for ucsd_id in list(book_id_to_db_id.keys()):
                if ucsd_id in book_id_to_isbn:
                    del book_id_to_db_id[ucsd_id]
            print(f"  {len(book_id_to_db_id):,} no-isbn books have review FK path.")

        ingest_reviews(
            conn, book_id_to_isbn, book_id_to_db_id,
            args.max_reviews, args.limit, args.dry_run,
        )

    if conn:
        conn.close()

    print("\nDone.")
    if args.dry_run:
        print("(dry run - nothing written)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
