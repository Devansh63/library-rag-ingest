"""
RAG Recommendation Pipeline — uses Groq API (free tier).

Search → top-K books → LLM prompt → natural language recommendation.
Falls back to structured list if no GROQ_API_KEY.
"""
from __future__ import annotations

import logging
import re

from app.core.config import settings
from app.services.query_classifier import classify_query
from app.services.search import hybrid_search, keyword_only_search

logger = logging.getLogger(__name__)

_MAX_CHAT_TURNS_CLIENT = 32  # alternating user/assistant (each counts as one entry)
_SEARCH_BLURB_MAX = 600
_LIBRARIAN_CHAT_SYSTEM = """\
You are a friendly, witty librarian chatting with a reader in a natural back-and-forth (like ChatGPT).

Use the ongoing conversation—you may clarify tastes, riff on moods, joke lightly, ask a short follow-up,
or tighten suggestions when they push back ("too long", "edgier", "more literary"). Keep replies readable:
short paragraphs unless they ask for depth.

IMPORTANT: Facts about specific books—titles, authors, stars, synopsis details—must come ONLY from the
"CATALOG excerpts" block attached to their latest message. If that block is thin or missing, say what you
know from the conversation and suggest they try a vaguer mood or genre; do not invent books.

When they ask for **more / different / other** titles, assume the new catalog block may contain books you
have not mentioned yet—do not insist the library “only” has what you listed in an earlier reply.

Respond as plain text/Markdown; no faux JSON."""


def _collapse_adjacent_roles(turns: list[dict[str, str]]) -> list[dict[str, str]]:
    """Groq prefers strict user/assistant alternation—merge back-to-back same roles."""
    out: list[dict[str, str]] = []
    for m in turns:
        if out and out[-1]["role"] == m["role"]:
            prev = out[-1]["content"]
            out[-1] = {"role": m["role"], "content": prev + "\n\n" + m["content"]}
        else:
            out.append(dict(m))
    return out


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


_FOLLOWUP_MORE = re.compile(
    r"\b("
    r"more|another|others?|different|else|again|next|fetch|show\s+me\s+more|"
    r"other\s+titles|more\s+books|keep\s+going|something\s+else"
    r")\b",
    re.IGNORECASE,
)


def _search_query_from_thread(messages: list[dict[str, str]]) -> str:
    """Build retrieval text from user lines, **newest first within the char budget** (old tail-truncation
    dropped early topics like “nature” after long threads)."""
    chunks: list[str] = []
    for m in messages:
        if m.get("role") == "user":
            t = str(m.get("content", "")).strip()
            if t:
                chunks.append(t)
    if not chunks:
        return ""
    picked_rev: list[str] = []
    budget = 0
    for t in reversed(chunks):
        add = len(t) + (2 if picked_rev else 0)
        if budget + add > _SEARCH_BLURB_MAX:
            break
        picked_rev.append(t)
        budget += add
    picked = list(reversed(picked_rev))
    joined = "\n".join(picked).strip()
    # Short “more / different” follow-ups: anchor to the previous substantive user ask.
    last = chunks[-1]
    if len(chunks) >= 2 and (len(last) < 140 or _FOLLOWUP_MORE.search(last)):
        prev = chunks[-2]
        if prev and prev not in last:
            joined = f"{prev}\nFollow-up: {last}"[:_SEARCH_BLURB_MAX]
    return joined or last


def _filter_excluded(books: list[dict], exclude: set[str]) -> list[dict]:
    if not exclude:
        return list(books)
    out: list[dict] = []
    for b in books:
        isbn = str(b.get("isbn13") or "").strip()
        if isbn and isbn in exclude:
            continue
        out.append(b)
    return out


def _enrich_with_keyword_diversity(
    classified, base: list[dict], exclude: set[str], want: int
) -> list[dict]:
    """If exclusions thin out results, pull extra BM25 rows and merge by id."""
    if len(base) >= want or not classified.cleaned:
        return base
    try:
        extra = keyword_only_search(classified.cleaned, limit=40)
    except Exception as e:
        logger.warning("keyword_only_search fill-in failed: %s", e)
        return base
    seen = {int(b["id"]) for b in base if b.get("id") is not None}
    merged = list(base)
    for b in extra:
        bid = b.get("id")
        if bid is None or int(bid) in seen:
            continue
        isbn = str(b.get("isbn13") or "").strip()
        if isbn and isbn in exclude:
            continue
        merged.append(b)
        seen.add(int(bid))
        if len(merged) >= want:
            break
    return merged


async def conversational_recommendations(
    messages: list[dict[str, str]],
    *,
    catalog_limit: int,
    exclude_isbn13: list[str] | None = None,
) -> dict:
    """Multi-turn librarian chat: fresh hybrid retrieval each user reply + Groq dialog."""

    sanitized: list[dict[str, str]] = []
    for m in messages:
        role = str(m.get("role", "")).lower().strip()
        content = str(m.get("content", "")).strip()
        if role not in ("user", "assistant") or not content:
            continue
        if len(content) > 12000:
            content = content[:12000] + "…"
        sanitized.append({"role": role, "content": content})

    sanitized = _collapse_adjacent_roles(sanitized)[-_MAX_CHAT_TURNS_CLIENT:]

    if not sanitized or sanitized[-1]["role"] != "user":
        return {
            "error": "invalid_messages",
            "detail": "Send a non-empty transcript ending with a user message.",
            "messages": sanitized,
            "assistant_reply": None,
            "books": [],
        }

    exclude_set = {str(x).strip() for x in (exclude_isbn13 or []) if str(x).strip()}
    search_q = _search_query_from_thread(sanitized)
    classified = classify_query(search_q[-500:] if search_q else "books")

    fuse_cap = min(
        56,
        settings.rag_top_k + max(12, len(exclude_set) * 2),
    )
    per_sig = min(100, 55 + len(exclude_set))
    books = hybrid_search(classified, limit=fuse_cap, per_signal_limit=per_sig)
    books = _filter_excluded(books, exclude_set)
    books = _enrich_with_keyword_diversity(
        classified, books, exclude_set, want=max(catalog_limit + 8, fuse_cap // 2)
    )

    plain_last = sanitized[-1]["content"]

    if not books:
        no_match_note = (
            "\n\n--- CATALOG excerpts (ground book facts here)\n---\n"
            "(No catalogue rows scored well this round—be honest, stay conversational, invite them to widen or tweak the vibe.)"
        )
        augmented = plain_last + no_match_note
        reply: str | None = None
        if settings.groq_api_key:
            try:
                reply = await _call_groq_multiturn(sanitized[:-1], augmented)
            except Exception as e:
                logger.error("Groq chat failed (empty catalog): %s", e)
        if reply is None:
            reply = (
                "I'm not pulling strong catalogue hits yet—tell me softer constraints "
                '(pace, comps, mood) or swap genre a bit and I\'ll hunt again!'
            )

        return {
            "messages": sanitized + [{"role": "assistant", "content": reply}],
            "assistant_reply": reply,
            "books": [],
            "query_type": classified.query_type,
        }

    # Keep excerpt block bounded so Groq still has room for long threads.
    ctx_books = books[: min(18, len(books))]
    context = _format_book_context(ctx_books)
    augmented = plain_last + (
        "\n\n--- CATALOG excerpts (ground book facts here)\n---\n" + context
    )

    if settings.groq_api_key:
        try:
            recommendation = await _call_groq_multiturn(sanitized[:-1], augmented)
        except Exception as e:
            logger.error("Groq chat RAG failed: %s", e)
            recommendation = _fallback_recommendation(books, plain_last)
    else:
        recommendation = _fallback_recommendation(books, plain_last)

    return {
        "messages": sanitized + [{"role": "assistant", "content": recommendation}],
        "assistant_reply": recommendation,
        "query_type": classified.query_type,
        "books": books[:catalog_limit],
    }


async def generate_recommendations(query: str, limit: int = 5) -> dict:
    """Single-shot RAG pipeline: classify → search → retrieve → generate."""
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


async def _call_groq_multiturn(
    prior_turns: list[dict[str, str]],
    final_user_payload: str,
) -> str:
    """Groq Chat Completions with full thread + augmented final user bubble."""
    from openai import AsyncOpenAI

    pt = list(prior_turns)
    while pt and pt[0]["role"] == "assistant":
        pt = pt[1:]

    client = AsyncOpenAI(
        api_key=settings.groq_api_key,
        base_url=settings.groq_base_url,
        timeout=90.0,
    )
    messages: list[dict[str, str]] = [{"role": "system", "content": _LIBRARIAN_CHAT_SYSTEM}]
    messages.extend(pt)
    messages.append({"role": "user", "content": final_user_payload})

    response = await client.chat.completions.create(
        model=settings.rag_model,
        messages=messages,
        max_tokens=1024,
        temperature=0.75,
    )
    raw = response.choices[0].message.content
    return (raw or "").strip() or "(No reply text returned.)"


async def _call_groq(query: str, context: str) -> str:
    """Call Groq API for RAG recommendation. Uses OpenAI-compatible format."""
    from openai import AsyncOpenAI

    client = AsyncOpenAI(
        api_key=settings.groq_api_key,
        base_url=settings.groq_base_url,
        timeout=90.0,
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
