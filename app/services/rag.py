"""
RAG Recommendation Pipeline — uses Groq API (free tier).

Search → top-K books → LLM prompt → natural language recommendation.
Falls back to structured list if no GROQ_API_KEY.
"""
from __future__ import annotations

import logging

from app.core.config import settings
from app.services.query_classifier import classify_query
from app.services.search import hybrid_search

logger = logging.getLogger(__name__)


def _format_book_context(books: list[dict]) -> str:
    parts = []
    for i, book in enumerate(books, 1):
        title = book.get("title", "Unknown")
        authors = ", ".join(book.get("authors") or ["Unknown"])
        genres = ", ".join(book.get("genres") or [])
        rating = book.get("goodreads_rating")
        num_ratings = book.get("num_ratings")
        synopsis = book.get("synopsis", "")
        if synopsis and len(synopsis) > 500:
            synopsis = synopsis[:500] + "..."
        rating_str = f"{rating:.1f}/5 ({num_ratings:,} ratings)" if rating else "No rating"
        parts.append(
            f"Book {i}: {title}\n"
            f"  Author(s): {authors}\n"
            f"  Genres: {genres or 'N/A'}\n"
            f"  Rating: {rating_str}\n"
            f"  Description: {synopsis or 'No description available.'}\n"
        )
    return "\n".join(parts)


def _build_rag_prompt(query: str, context: str) -> str:
    return f"""You are a knowledgeable and friendly librarian helping a reader find their next book.
Based on the reader's request and the book catalog information below, provide personalized recommendations.

READER'S REQUEST: {query}

CATALOG RESULTS:
{context}

INSTRUCTIONS:
- Recommend 3-5 books from the catalog results above that best match the reader's request.
- For each recommendation, explain in 1-2 sentences WHY this book matches what they're looking for.
- Be conversational and enthusiastic but not over-the-top.
- If the catalog results don't perfectly match the request, acknowledge that and still suggest the closest matches.
- Reference specific details from the book descriptions to show genuine understanding.

Provide your recommendations below:"""


async def generate_recommendations(query: str, limit: int = 5) -> dict:
    """Full RAG pipeline: classify → search → retrieve → generate."""
    classified = classify_query(query)
    books = hybrid_search(classified, limit=settings.rag_top_k)

    if not books:
        return {
            "query": query,
            "books": [],
            "recommendation": "I couldn't find any books matching your request. Try rephrasing your query.",
            "query_type": classified.query_type,
        }

    context = _format_book_context(books)
    recommendation = None

    if settings.groq_api_key:
        try:
            recommendation = await _call_groq(query, context)
        except Exception as e:
            logger.error("Groq RAG call failed: %s", e)
            recommendation = _fallback_recommendation(books, query)
    else:
        recommendation = _fallback_recommendation(books, query)

    return {
        "query": query,
        "recommendation": recommendation,
        "query_type": classified.query_type,
        "books": books[:limit],
    }


async def _call_groq(query: str, context: str) -> str:
    """Call Groq API for RAG recommendation. Uses OpenAI-compatible format."""
    from openai import AsyncOpenAI

    client = AsyncOpenAI(
        api_key=settings.groq_api_key,
        base_url="https://api.groq.com/openai/v1",
    )
    prompt = _build_rag_prompt(query, context)

    response = await client.chat.completions.create(
        model=settings.rag_model,
        messages=[
            {"role": "system", "content": "You are a helpful librarian. Provide book recommendations based on catalog search results."},
            {"role": "user", "content": prompt},
        ],
        max_tokens=800,
        temperature=0.7,
    )
    return response.choices[0].message.content


def _fallback_recommendation(books: list[dict], query: str) -> str:
    """Structured recommendation without an LLM."""
    if not books:
        return "No matching books found."
    lines = [f'Based on your search for "{query}", here are my top picks:\n']
    for i, book in enumerate(books[:5], 1):
        title = book.get("title", "Unknown")
        authors = ", ".join(book.get("authors") or ["Unknown"])
        rating = book.get("goodreads_rating")
        rating_str = f" ({rating:.1f}/5)" if rating else ""
        synopsis = book.get("synopsis", "")
        blurb = (synopsis[:150] + "...") if synopsis and len(synopsis) > 150 else synopsis
        lines.append(f"{i}. **{title}** by {authors}{rating_str}")
        if blurb:
            lines.append(f"   {blurb}")
        lines.append("")
    return "\n".join(lines)
