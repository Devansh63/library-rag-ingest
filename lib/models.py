"""
Pydantic models for the ingestion pipeline.

These mirror the actual DB_1 schema (as of 2026-04-10), not the paper's
Figure 2. The relevant drifts are:

- `books.source` is TEXT in the DB but was described as varchar[] in the
  paper. We store a single source label per row (e.g. "goodreads_bbe").
- `books.metadata_embedding` and `books.review_embedding` are vector(768)
  columns left NULL by ingestion; a teammate fills them in a later pass.
- `books.isbn13` and `books.isbn10` are both nullable, because the CMU
  summary dataset often lacks ISBNs entirely.

Only the books and reviews models are defined here. Inventory, borrows,
users, and search_logs are outside Devansh's ingestion scope.
"""

from __future__ import annotations

from datetime import date
from enum import Enum
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


class BookSource(str, Enum):
    """Label written into books.source to mark which dataset a row came from.

    Keep these values stable; downstream queries and de-dup logic will key
    off them.
    """

    GOODREADS_BBE = "goodreads_bbe"       # Zenodo Best Books Ever dump
    CMU_SUMMARIES = "cmu_summaries"       # CMU Book Summary Dataset
    UCSD_GRAPH = "ucsd_graph"             # UCSD Goodreads Book Graph
    ISBNDB = "isbndb"                     # Enrichment-only, not a primary row
    MERGED = "merged"                     # Row was merged across 2+ sources


class ReviewSource(str, Enum):
    """Label written into reviews.source."""

    UCSD_GRAPH = "ucsd_graph"
    USER_SUBMITTED = "user_submitted"
    ISBNDB = "isbndb"


class BookRow(BaseModel):
    """One row destined for the `books` table.

    All fields except `title` and `source` are optional because each upstream
    dataset populates a different subset. The ingestion merge step fills in
    missing fields from other sources when it can.
    """

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    isbn13: Optional[str] = Field(default=None, max_length=13)
    isbn10: Optional[str] = Field(default=None, max_length=10)
    title: str = Field(min_length=1)
    authors: list[str] = Field(default_factory=list)
    publisher: Optional[str] = None
    publish_date: Optional[date] = None
    first_publish_date: Optional[date] = None
    genres: list[str] = Field(default_factory=list)
    subjects: list[str] = Field(default_factory=list)
    language: Optional[str] = None  # Leave None when unknown; DB defaults to 'en'.
    pages: Optional[int] = Field(default=None, ge=0)
    edition: Optional[str] = None
    series: Optional[str] = None
    short_description: Optional[str] = None
    synopsis: Optional[str] = None
    plot_summary: Optional[str] = None
    cover_image_url: Optional[str] = None
    goodreads_rating: Optional[float] = Field(default=None, ge=0.0, le=5.0)
    num_ratings: Optional[int] = Field(default=None, ge=0)
    awards: list[str] = Field(default_factory=list)
    source: BookSource
    # Quality flags set by the UCSD ingest when a row passes minimum bars but
    # has fixable issues. NULL means "not evaluated" (non-UCSD sources). An
    # empty list after UCSD ingest means "evaluated and clean". Non-empty means
    # the row needs ISBNdb enrichment or manual review before it can be trusted.
    # Intentionally excluded from _none_to_empty so None stays None for rows
    # from other sources that were never evaluated.
    cleaning_flags: list[str] = Field(default_factory=list)
    # UCSD's internal book_id, stored for all ucsd_graph rows so we can look
    # up books.id after a bulk insert and link reviews for no-isbn books.
    ucsd_book_id: Optional[str] = None

    @field_validator("isbn13", "isbn10", mode="before")
    @classmethod
    def _blank_to_none(cls, value: object) -> object:
        # Upstream datasets use empty strings to mean "missing". Normalize so
        # the DB gets NULL instead of '' and deduplication works cleanly.
        if isinstance(value, str) and value.strip() == "":
            return None
        return value

    @field_validator("authors", "genres", "subjects", "awards", mode="before")
    @classmethod
    def _none_to_empty(cls, value: object) -> object:
        # None -> [] so the DB always gets a real text[] value, never NULL
        # for array columns. Easier to query downstream.
        if value is None:
            return []
        return value


class ReviewType(str, Enum):
    """Distinguishes user-written reviews from editorial/professional ones.

    - USER: community reviews from UCSD Goodreads dataset
    - EDITORIAL: professional reviews from publishers (e.g. Kirkus, ISBNdb)
      These have no star rating and user_id holds the publication name.
    """

    USER = "user"
    EDITORIAL = "editorial"


class ReviewRow(BaseModel):
    """One row destined for the `reviews` table."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    isbn13: Optional[str] = Field(default=None, max_length=13)
    # Direct FK to books.id. Set for no-isbn books where isbn13 is unavailable.
    # Exactly one of isbn13 or book_id should be set per row.
    book_id: Optional[int] = None
    user_id: Optional[str] = None
    rating: Optional[int] = Field(default=None, ge=0, le=5)
    review_text: Optional[str] = None
    date_posted: Optional[date] = None
    spoiler_flag: bool = False
    source: ReviewSource = ReviewSource.UCSD_GRAPH
    review_type: ReviewType = ReviewType.USER
