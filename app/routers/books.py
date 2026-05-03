"""Book detail, review, and recommendation endpoints."""
from __future__ import annotations

from typing import Literal

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from app.core.db import execute_query
from app.services.rag import conversational_recommendations, generate_recommendations

router = APIRouter(tags=["books"])


class ChatTurn(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(..., min_length=1, max_length=16000)


class RecommendChatBody(BaseModel):
    """Conversation turns (must end with a user message)."""

    messages: list[ChatTurn] = Field(..., min_length=1)
    limit: int = Field(default=5, ge=1, le=10)
    # When the reader asks for “more / different” picks, pass ISBNs already shown so retrieval can widen.
    exclude_isbn13: list[str] = Field(default_factory=list, max_length=80)


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
    """One-shot RAG recommendations (backward compatible)."""
    return await generate_recommendations(q, limit=limit)


@router.post("/recommend/chat")
async def recommend_chat(body: RecommendChatBody):
    """Multi-turn librarian chat: retrieval each turn + Groq dialog."""
    payload = [{"role": t.role, "content": t.content} for t in body.messages]
    exclude = [x.strip() for x in body.exclude_isbn13 if x and str(x).strip()]
    result = await conversational_recommendations(
        payload, catalog_limit=body.limit, exclude_isbn13=exclude
    )
    if result.get("error") == "invalid_messages":
        raise HTTPException(status_code=400, detail=result.get("detail", "Invalid transcript."))
    return result
