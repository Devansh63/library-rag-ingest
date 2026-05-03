"""Analytics endpoints: stats, genres, popular."""
from __future__ import annotations
from fastapi import APIRouter, Query
from app.core.db import execute_query

router = APIRouter(prefix="/analytics", tags=["analytics"])


@router.get("/popular")
def popular_books(limit: int = Query(default=20, ge=1, le=100), period_days: int = Query(default=30, ge=1, le=365)):
    """Most-borrowed books in the given time period."""
    return execute_query("""
        SELECT b.isbn13, b.title, b.authors, b.genres, b.goodreads_rating, b.cover_image_url,
            COUNT(br.id) AS borrow_count
        FROM borrows br JOIN inventory i ON i.id = br.inventory_id JOIN books b ON b.isbn13 = i.isbn13
        WHERE br.borrow_date >= CURRENT_DATE - %(days)s * INTERVAL '1 day'
        GROUP BY b.isbn13, b.title, b.authors, b.genres, b.goodreads_rating, b.cover_image_url
        ORDER BY borrow_count DESC LIMIT %(limit)s
    """, {"days": period_days, "limit": limit})


@router.get("/genres")
def genre_distribution(limit: int = Query(default=30, ge=1, le=100)):
    """Top genres by book count with average rating."""
    return execute_query("""
        SELECT genre, COUNT(*) AS book_count, ROUND(AVG(b.goodreads_rating)::numeric, 2) AS avg_rating
        FROM books b, unnest(b.genres) AS genre
        WHERE b.genres IS NOT NULL AND array_length(b.genres, 1) > 0
        GROUP BY genre ORDER BY book_count DESC LIMIT %(limit)s
    """, {"limit": limit})


@router.get("/stats")
def database_stats():
    """Overall database statistics."""
    stats = {}
    book_rows = execute_query("""
        SELECT COUNT(*) AS total_books,
            COUNT(*) FILTER (WHERE isbn13 IS NOT NULL) AS with_isbn,
            COUNT(*) FILTER (WHERE metadata_embedding IS NOT NULL) AS with_embeddings,
            COUNT(*) FILTER (WHERE synopsis IS NOT NULL) AS with_synopsis,
            COUNT(*) FILTER (WHERE array_length(authors, 1) > 0) AS with_authors,
            ROUND(AVG(goodreads_rating)::numeric, 2) AS avg_rating
        FROM books
    """)
    if book_rows:
        stats["books"] = book_rows[0]
    review_rows = execute_query("""
        SELECT COUNT(*) AS total_reviews,
            ROUND(AVG(rating)::numeric, 2) AS avg_rating
        FROM reviews
    """)
    if review_rows:
        stats["reviews"] = review_rows[0]
    source_rows = execute_query("SELECT source, COUNT(*) AS count FROM books GROUP BY source ORDER BY count DESC")
    stats["sources"] = {row["source"]: row["count"] for row in source_rows}
    return stats
