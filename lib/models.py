"""
Pydantic models for the ingestion pipeline - mirror the books and reviews tables.

Notable schema quirks:
  - isbn13/isbn10 are nullable (CMU dataset often has no ISBNs)
  - metadata_embedding and review_embedding are populated by the embed script, not here
  - cleaning_flags NULL means "not evaluated" (non-UCSD rows); empty list means "clean"
"""

from __future__ import annotations

from datetime import date
from enum import Enum
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


class BookSource(str, Enum):
    GOODREADS_BBE = "goodreads_bbe"
    CMU_SUMMARIES = "cmu_summaries"
    UCSD_GRAPH    = "ucsd_graph"
    ISBNDB        = "isbndb"
    MERGED        = "merged"


class ReviewSource(str, Enum):
    UCSD_GRAPH      = "ucsd_graph"
    USER_SUBMITTED  = "user_submitted"
    ISBNDB          = "isbndb"


class BookRow(BaseModel):
    """One row for the `books` table. Most fields are optional - each source
    populates a different subset and the merge step fills in what it can."""

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
    language: Optional[str] = None
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
    # NULL = not evaluated; empty list = evaluated and clean; non-empty = needs review
    cleaning_flags: list[str] = Field(default_factory=list)
    # Stored for ucsd_graph rows to let the ingestor map UCSD ids to internal book ids
    # when linking no-isbn reviews.
    ucsd_book_id: Optional[str] = None

    @field_validator("isbn13", "isbn10", mode="before")
    @classmethod
    def _blank_to_none(cls, value: object) -> object:
        if isinstance(value, str) and value.strip() == "":
            return None
        return value

    @field_validator("authors", "genres", "subjects", "awards", mode="before")
    @classmethod
    def _none_to_empty(cls, value: object) -> object:
        if value is None:
            return []
        return value


class ReviewType(str, Enum):
    USER       = "user"
    EDITORIAL  = "editorial"


class ReviewRow(BaseModel):
    """One row for the `reviews` table.

    Exactly one of isbn13 or book_id should be set - isbn13 for books that
    have one, book_id for no-isbn UCSD books.
    """

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    isbn13: Optional[str] = Field(default=None, max_length=13)
    book_id: Optional[int] = None
    user_id: Optional[str] = None
    rating: Optional[int] = Field(default=None, ge=0, le=5)
    review_text: Optional[str] = None
    date_posted: Optional[date] = None
    spoiler_flag: bool = False
    source: ReviewSource = ReviewSource.UCSD_GRAPH
    review_type: ReviewType = ReviewType.USER
