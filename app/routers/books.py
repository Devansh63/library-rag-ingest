"""Book detail, review, and recommendation endpoints."""
from __future__ import annotations
from fastapi import APIRouter, Query, HTTPException
from app.core.db import execute_query
from app.services.rag import generate_recommendations

router = APIRouter(tags=["books"])


@router.get("/books/{isbn13}")
def get_book(isbn13: str):
    rows = execute_query("""
        SELECT id, isbn13, isbn10, title, authors, publisher, publish_date,
            first_publish_date, genres, subjects, language, pages, edition,
            series, short_description, synopsis, plot_summary, cover_image_url,
            goodreads_rating, num_ratings, awards, source
        FROM books WHERE isbn13 = %(isbn13)s
    """, {"isbn13": isbn13})
    if not rows:
        raise HTTPException(status_code=404, detail="Book not found")
    return rows[0]


@router.get("/books/{isbn13}/reviews")
def get_book_reviews(isbn13: str, limit: int = Query(default=20, ge=1, le=100),
                     min_rating: int = Query(default=None, ge=1, le=5)):
    conditions = ["(r.isbn13 = %(isbn13)s OR r.book_id = (SELECT id FROM books WHERE isbn13 = %(isbn13)s LIMIT 1))"]
    params: dict = {"isbn13": isbn13, "limit": limit}
    if min_rating is not None:
        conditions.append("r.rating >= %(min_rating)s")
        params["min_rating"] = min_rating
    sql = f"""
        SELECT r.id, r.user_id, r.rating, r.review_text, r.date_posted,
            r.spoiler_flag, r.source, r.review_type
        FROM reviews r WHERE {' AND '.join(conditions)}
        ORDER BY r.rating DESC NULLS LAST LIMIT %(limit)s
    """
    return execute_query(sql, params)


@router.get("/recommend")
async def recommend(q: str = Query(..., min_length=3), limit: int = Query(default=5, ge=1, le=10)):
    """RAG-powered book recommendations."""
    return await generate_recommendations(q, limit=limit)
