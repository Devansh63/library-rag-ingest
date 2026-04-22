"""
Database helpers for ingesting into the DB_1 Neon Postgres instance.

The two entry points most scripts will use are:

- `get_connection()`: read `DATABASE_URL_1` from the environment and open
  one psycopg2 connection. We intentionally do not use connection pooling
  for ingestion runs; a single long-lived connection fits the workflow and
  avoids surprising Neon's free-tier concurrency limits.

- `bulk_insert_books()` and `bulk_insert_reviews()`: wrap
  `psycopg2.extras.execute_values` with `page_size=200`, matching the
  strategy the milestone paper calls for. Batching cuts round-trip latency
  to Neon from roughly 80 seconds per 16k rows to under a second.

All inserts use ON CONFLICT DO NOTHING on isbn13 where possible so repeated
ingestion runs are idempotent and safe to re-run while debugging. The first
source (Zenodo) lays down the ISBN keys; subsequent sources upsert their
extra fields via ON CONFLICT DO UPDATE in later helpers (not yet written).
"""

from __future__ import annotations

import os
from collections.abc import Iterable
from contextlib import contextmanager
from typing import Any

import psycopg2
from dotenv import load_dotenv
from psycopg2.extensions import connection as PgConnection
from psycopg2.extras import execute_values

from .models import BookRow, ReviewRow


# Load .env once at module import so every script that imports lib.db picks
# up DATABASE_URL_1, DATABASE_URL_2, and ISBNDB_API_KEY without each script
# needing its own load_dotenv() call.
load_dotenv()


BULK_PAGE_SIZE = 200  # Matches the paper (Section 4.4).


def get_connection(env_var: str = "DATABASE_URL_1") -> PgConnection:
    """Open a new psycopg2 connection using the env var specified.

    Defaults to DATABASE_URL_1 because that is the chosen ingestion target
    as of 2026-04-10. Override with `env_var="DATABASE_URL_2"` only for
    inspection or scratch work.
    """
    url = os.environ.get(env_var)
    if not url:
        # Fail loudly - silently falling through to "localhost with no creds"
        # would be much worse than an immediate crash.
        raise RuntimeError(
            f"{env_var} is not set. Populate it in .env before running ingestion."
        )
    # Neon auto-suspends when idle; a generous connect timeout lets the DB
    # wake up instead of failing on a cold start.
    return psycopg2.connect(url, connect_timeout=30)


@contextmanager
def transaction(conn: PgConnection):
    """Yield a cursor inside an explicit transaction.

    psycopg2 opens a transaction implicitly on first statement, but wrapping
    it in a context manager makes the commit/rollback behavior obvious to
    anyone reading the ingestion scripts.
    """
    try:
        with conn.cursor() as cursor:
            yield cursor
        conn.commit()
    except Exception:
        conn.rollback()
        raise


# Column order must match the VALUES tuples we build in the bulk insert.
# Keeping this as a module constant so the tuple-builder and the SQL stay
# in lock-step; mismatches here are the single most painful bug class in
# execute_values code.
_BOOK_COLUMNS = (
    "isbn13",
    "isbn10",
    "title",
    "authors",
    "publisher",
    "publish_date",
    "first_publish_date",
    "genres",
    "subjects",
    "language",
    "pages",
    "edition",
    "series",
    "short_description",
    "synopsis",
    "plot_summary",
    "cover_image_url",
    "goodreads_rating",
    "num_ratings",
    "awards",
    "source",
    "cleaning_flags",
    "ucsd_book_id",
)


def _book_to_tuple(book: BookRow) -> tuple[Any, ...]:
    """Convert a BookRow into the exact tuple shape the insert expects."""
    return (
        book.isbn13,
        book.isbn10,
        book.title,
        book.authors,               # Postgres text[] accepts Python list.
        book.publisher,
        book.publish_date,
        book.first_publish_date,
        book.genres,
        book.subjects,
        book.language,              # None -> DB default 'en'.
        book.pages,
        book.edition,
        book.series,
        book.short_description,
        book.synopsis,
        book.plot_summary,
        book.cover_image_url,
        book.goodreads_rating,
        book.num_ratings,
        book.awards,
        book.source.value,
        # Empty list -> NULL so non-UCSD rows stay NULL ("not evaluated").
        # Non-empty list -> Postgres text[] array ("has cleaning issues").
        book.cleaning_flags or None,
        book.ucsd_book_id,
    )


def ensure_ucsd_book_id_column(conn: PgConnection) -> None:
    """Add books.ucsd_book_id if it doesn't exist yet. Idempotent."""
    with conn.cursor() as cur:
        cur.execute(
            "ALTER TABLE books ADD COLUMN IF NOT EXISTS ucsd_book_id text"
        )
    conn.commit()


def ensure_review_book_id_column(conn: PgConnection) -> None:
    """Add reviews.book_id integer column if it doesn't exist yet. Idempotent.

    This is the second FK path for linking reviews to no-isbn books.
    isbn-bearing books still use reviews.isbn13 as before.
    """
    with conn.cursor() as cur:
        cur.execute(
            "ALTER TABLE reviews ADD COLUMN IF NOT EXISTS book_id integer"
        )
    conn.commit()


def ensure_cleaning_flags_column(conn: PgConnection) -> None:
    """Add the cleaning_flags text[] column to books if it doesn't exist yet.

    Safe to call on every run - uses ADD COLUMN IF NOT EXISTS so it is a
    no-op once the column is present. Calling this before any UCSD ingest
    avoids the need to run a separate migration script.
    """
    with conn.cursor() as cur:
        cur.execute(
            "ALTER TABLE books ADD COLUMN IF NOT EXISTS cleaning_flags text[]"
        )
    conn.commit()


def bulk_insert_books(
    conn: PgConnection,
    books: Iterable[BookRow],
    *,
    on_conflict: str = "DO NOTHING",
) -> int:
    """Insert many books in one batched round trip. Returns row count inserted.

    `on_conflict` is interpolated raw into the SQL so callers can switch to
    DO UPDATE when doing a multi-source enrichment pass. It is not sourced
    from user input, so the raw interpolation is safe here.
    """
    rows = [_book_to_tuple(b) for b in books]
    if not rows:
        return 0

    columns_sql = ", ".join(_BOOK_COLUMNS)
    sql = (
        f"INSERT INTO books ({columns_sql}) VALUES %s "
        f"ON CONFLICT (isbn13) {on_conflict}"
    )

    with transaction(conn) as cursor:
        # NOTE: ON CONFLICT (isbn13) requires a unique index on isbn13. If
        # the DB does not have one yet, this will error - handle that in the
        # caller by either creating the index or switching to a different
        # conflict target.
        execute_values(cursor, sql, rows, page_size=BULK_PAGE_SIZE)
        # execute_values does not return a row count for batched inserts,
        # so we report the input count as an upper bound. Caller can diff
        # against a SELECT COUNT(*) if an exact figure is needed.
    return len(rows)


_REVIEW_COLUMNS = (
    "isbn13",
    "book_id",
    "user_id",
    "rating",
    "review_text",
    "date_posted",
    "spoiler_flag",
    "source",
    "review_type",
)


def _review_to_tuple(review: ReviewRow) -> tuple[Any, ...]:
    return (
        review.isbn13,
        review.book_id,
        review.user_id,
        review.rating,
        review.review_text,
        review.date_posted,
        review.spoiler_flag,
        review.source.value,
        review.review_type.value,
    )


def bulk_insert_reviews(conn: PgConnection, reviews: Iterable[ReviewRow]) -> int:
    """Insert many reviews in one batched round trip."""
    rows = [_review_to_tuple(r) for r in reviews]
    if not rows:
        return 0

    columns_sql = ", ".join(_REVIEW_COLUMNS)
    # Reviews have no natural unique key beyond (isbn13, user_id, date), and
    # the schema has no unique constraint on reviews, so we do a plain INSERT.
    # This means re-running an ingestion will duplicate reviews; callers
    # should TRUNCATE reviews first or add a dedupe check during a re-run.
    sql = f"INSERT INTO reviews ({columns_sql}) VALUES %s"

    with transaction(conn) as cursor:
        execute_values(cursor, sql, rows, page_size=BULK_PAGE_SIZE)
    return len(rows)
