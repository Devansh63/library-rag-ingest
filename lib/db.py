"""
DB helpers for the ingestion pipeline (DATABASE_URL_1 / Neon Postgres).

Main entry points:
  get_connection()         -- open one psycopg2 connection
  bulk_insert_books()      -- batched execute_values insert for books
  bulk_insert_reviews()    -- batched execute_values insert for reviews
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

load_dotenv()

BULK_PAGE_SIZE = 200


def get_connection(env_var: str = "DATABASE_URL_1") -> PgConnection:
    url = os.environ.get(env_var)
    if not url:
        raise RuntimeError(
            f"{env_var} is not set. Populate it in .env before running ingestion."
        )
    # connect_timeout lets Neon wake from auto-suspend instead of failing cold.
    return psycopg2.connect(url, connect_timeout=30)


@contextmanager
def transaction(conn: PgConnection):
    try:
        with conn.cursor() as cursor:
            yield cursor
        conn.commit()
    except Exception:
        conn.rollback()
        raise


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
    return (
        book.isbn13,
        book.isbn10,
        book.title,
        book.authors,
        book.publisher,
        book.publish_date,
        book.first_publish_date,
        book.genres,
        book.subjects,
        book.language,
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
        book.cleaning_flags or None,
        book.ucsd_book_id,
    )


def ensure_ucsd_book_id_column(conn: PgConnection) -> None:
    """Add books.ucsd_book_id if it doesn't exist yet."""
    with conn.cursor() as cur:
        cur.execute("ALTER TABLE books ADD COLUMN IF NOT EXISTS ucsd_book_id text")
    conn.commit()


def ensure_review_book_id_column(conn: PgConnection) -> None:
    """Add reviews.book_id integer column if it doesn't exist yet."""
    with conn.cursor() as cur:
        cur.execute("ALTER TABLE reviews ADD COLUMN IF NOT EXISTS book_id integer")
    conn.commit()


def ensure_cleaning_flags_column(conn: PgConnection) -> None:
    """Add books.cleaning_flags text[] column if it doesn't exist yet."""
    with conn.cursor() as cur:
        cur.execute("ALTER TABLE books ADD COLUMN IF NOT EXISTS cleaning_flags text[]")
    conn.commit()


def bulk_insert_books(
    conn: PgConnection,
    books: Iterable[BookRow],
    *,
    on_conflict: str = "DO NOTHING",
) -> int:
    """Insert books in one batched round trip. Returns input count (upper bound on inserts)."""
    rows = [_book_to_tuple(b) for b in books]
    if not rows:
        return 0

    columns_sql = ", ".join(_BOOK_COLUMNS)
    sql = (
        f"INSERT INTO books ({columns_sql}) VALUES %s "
        f"ON CONFLICT (isbn13) {on_conflict}"
    )

    with transaction(conn) as cursor:
        execute_values(cursor, sql, rows, page_size=BULK_PAGE_SIZE)
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
    """Insert reviews in one batched round trip.

    No unique constraint on reviews, so re-running will duplicate rows.
    Truncate first or add a dedupe check if re-running an ingest.
    """
    rows = [_review_to_tuple(r) for r in reviews]
    if not rows:
        return 0

    columns_sql = ", ".join(_REVIEW_COLUMNS)
    sql = f"INSERT INTO reviews ({columns_sql}) VALUES %s"

    with transaction(conn) as cursor:
        execute_values(cursor, sql, rows, page_size=BULK_PAGE_SIZE)
    return len(rows)
