"""
Hybrid Search — BM25 + Metadata Cosine + Review Cosine → RRF Fusion.

Three retrieval signals, merged via Reciprocal Rank Fusion (RRF).
Query embeddings are generated via the HuggingFace Inference API
(BAAI/bge-base-en-v1.5) — no local model loaded, no GPU/RAM required.
"""
from __future__ import annotations

import logging
import math
import os

import httpx

from app.core.config import settings
from app.core.db import execute_query
from app.services.query_classifier import ClassifiedQuery

logger = logging.getLogger(__name__)

# Same model used to generate the stored embeddings - must not change.
# Use the HF router endpoint - api-inference.huggingface.co/models/ returns 404 for BGE.
_HF_API_URL = "https://router.huggingface.co/hf-inference/models/BAAI/bge-base-en-v1.5"


def embed_query(text: str) -> list[float]:
    """Encode a query string into a 768-dim vector via HuggingFace Inference API.

    Uses the same BAAI/bge-base-en-v1.5 model that generated the stored
    embeddings, so cosine similarity is valid. Normalizes to unit vector
    to match how stored embeddings were generated.
    """
    token = os.environ.get("HF_TOKEN", "")
    prefixed = "Represent this sentence for searching relevant passages: " + text

    response = httpx.post(
        _HF_API_URL,
        headers={"Authorization": f"Bearer {token}"},
        # wait_for_model=True: if HF's copy is cold, wait instead of returning 503.
        json={"inputs": prefixed, "options": {"wait_for_model": True}},
        timeout=30.0,
    )
    response.raise_for_status()

    # Router endpoint returns a flat list [float, ...] for a single string input.
    # Unwrap if nested just in case.
    raw = response.json()
    vec: list[float] = raw[0] if isinstance(raw[0], list) else raw

    # Normalize to unit vector - matches normalize_embeddings=True used during ingest.
    norm = math.sqrt(sum(x * x for x in vec))
    if norm > 0:
        vec = [x / norm for x in vec]
    return vec


# ---------------------------------------------------------------------------
# Individual retrieval signals
# ---------------------------------------------------------------------------

def bm25_search(query: str, limit: int = 50) -> list[dict]:
    """BM25 keyword search using Postgres full-text search.

    Only rows with ``search_tsv`` set participate in FTS (GIN index, no NULL
    scan). ``MATERIALIZED`` CTEs keep the fallback branches from re-planning
    the FTS leg. ISBN matches run in a separate cheap leg so we skip the
    ILIKE/unnest scan when an ISBN hit exists. Run
    ``scripts/backfill_search_tsv.py`` for full coverage.
    """
    sql = """
        WITH q AS (
            SELECT plainto_tsquery('english', %(query)s) AS tq
        ),
        fts AS MATERIALIZED (
            SELECT
                b.id, b.isbn13, b.title, b.authors, b.genres, b.synopsis,
                b.goodreads_rating, b.num_ratings, b.cover_image_url,
                ts_rank_cd(b.search_tsv, q.tq) AS rank
            FROM books b
            INNER JOIN q ON (b.search_tsv @@ q.tq)
            WHERE b.search_tsv IS NOT NULL
            ORDER BY rank DESC
            LIMIT %(limit)s
        ),
        fts_ok AS (
            SELECT * FROM fts WHERE rank > 0
        ),
        isbn_fb AS MATERIALIZED (
            SELECT
                b.id, b.isbn13, b.title, b.authors, b.genres, b.synopsis,
                b.goodreads_rating, b.num_ratings, b.cover_image_url,
                0.01::double precision AS rank
            FROM books b
            WHERE NOT EXISTS (SELECT 1 FROM fts)
              AND (b.isbn13 = %(query)s OR b.isbn10 = %(query)s)
            ORDER BY b.num_ratings DESC NULLS LAST
            LIMIT %(limit)s
        ),
        like_fb AS (
            SELECT
                b.id, b.isbn13, b.title, b.authors, b.genres, b.synopsis,
                b.goodreads_rating, b.num_ratings, b.cover_image_url,
                0.01::double precision AS rank
            FROM books b
            WHERE NOT EXISTS (SELECT 1 FROM fts)
              AND NOT EXISTS (SELECT 1 FROM isbn_fb)
              AND (
                  b.title ILIKE '%%' || %(query)s || '%%'
                  OR EXISTS (
                      SELECT 1 FROM unnest(b.authors) AS a
                      WHERE a ILIKE '%%' || %(query)s || '%%'
                  )
              )
            ORDER BY b.num_ratings DESC NULLS LAST
            LIMIT %(limit)s
        )
        SELECT * FROM fts_ok
        UNION ALL
        SELECT * FROM isbn_fb
        UNION ALL
        SELECT * FROM like_fb
    """
    return execute_query(sql, {"query": query, "limit": limit})


def metadata_cosine_search(query_embedding: list[float], limit: int = 50) -> list[dict]:
    """Semantic search against books.metadata_embedding."""
    embedding_str = "[" + ",".join(str(x) for x in query_embedding) + "]"
    sql = """
        SELECT
            b.id, b.isbn13, b.title, b.authors, b.genres, b.synopsis,
            b.goodreads_rating, b.num_ratings, b.cover_image_url,
            1 - (b.metadata_embedding <=> %(embedding)s::vector) AS similarity
        FROM books b
        WHERE b.metadata_embedding IS NOT NULL
        ORDER BY b.metadata_embedding <=> %(embedding)s::vector
        LIMIT %(limit)s
    """
    return execute_query(sql, {"embedding": embedding_str, "limit": limit})


def review_cosine_search(query_embedding: list[float], limit: int = 50) -> list[dict]:
    """Semantic search against review embeddings (pre-aggregated or runtime)."""
    embedding_str = "[" + ",".join(str(x) for x in query_embedding) + "]"

    # Check if books.review_embedding has data
    check = execute_query(
        "SELECT EXISTS(SELECT 1 FROM books WHERE review_embedding IS NOT NULL LIMIT 1) AS has"
    )
    has_pre_aggregated = check and check[0].get("has", False)

    if has_pre_aggregated:
        sql = """
            SELECT
                b.id, b.isbn13, b.title, b.authors, b.genres, b.synopsis,
                b.goodreads_rating, b.num_ratings, b.cover_image_url,
                1 - (b.review_embedding <=> %(embedding)s::vector) AS similarity
            FROM books b
            WHERE b.review_embedding IS NOT NULL
            ORDER BY b.review_embedding <=> %(embedding)s::vector
            LIMIT %(limit)s
        """
        return execute_query(sql, {"embedding": embedding_str, "limit": limit})

    # Fallback: runtime aggregation from reviews table
    sql = """
        SELECT
            b.id, b.isbn13, b.title, b.authors, b.genres, b.synopsis,
            b.goodreads_rating, b.num_ratings, b.cover_image_url,
            1 - (avg_emb.avg_vec <=> %(embedding)s::vector) AS similarity
        FROM (
            SELECT
                COALESCE(r.isbn13, b2.isbn13) AS book_isbn,
                AVG(r.review_embedding)::vector(768) AS avg_vec
            FROM reviews r
            LEFT JOIN books b2 ON r.book_id = b2.id
            WHERE r.review_embedding IS NOT NULL
            GROUP BY COALESCE(r.isbn13, b2.isbn13)
            HAVING COALESCE(r.isbn13, b2.isbn13) IS NOT NULL
        ) avg_emb
        JOIN books b ON b.isbn13 = avg_emb.book_isbn
        ORDER BY avg_emb.avg_vec <=> %(embedding)s::vector
        LIMIT %(limit)s
    """
    return execute_query(sql, {"embedding": embedding_str, "limit": limit})


# ---------------------------------------------------------------------------
# RRF Fusion
# ---------------------------------------------------------------------------

def reciprocal_rank_fusion(
    ranked_lists: list[tuple[list[dict], float]],
    k: int = 60,
    limit: int = 20,
) -> list[dict]:
    """Merge ranked lists via RRF. Each entry is (results, weight)."""
    scores: dict[int, float] = {}
    book_data: dict[int, dict] = {}

    for results, weight in ranked_lists:
        if not results:
            continue
        for rank, doc in enumerate(results, start=1):
            book_id = doc["id"]
            scores[book_id] = scores.get(book_id, 0.0) + weight / (k + rank)
            if book_id not in book_data:
                book_data[book_id] = doc

    sorted_ids = sorted(scores.keys(), key=lambda bid: scores[bid], reverse=True)

    results = []
    for book_id in sorted_ids[:limit]:
        doc = book_data[book_id]
        doc["rrf_score"] = round(scores[book_id], 6)
        results.append(doc)
    return results


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def hybrid_search(classified: ClassifiedQuery, limit: int = 20, per_signal_limit: int = 50) -> list[dict]:
    """Full hybrid search pipeline."""
    ranked_lists: list[tuple[list[dict], float]] = []

    if classified.weight_keyword > 0:
        try:
            ranked_lists.append((bm25_search(classified.cleaned, per_signal_limit), classified.weight_keyword))
        except Exception as e:
            logger.warning("BM25 search failed: %s", e)

    query_embedding = None
    if classified.weight_metadata > 0 or classified.weight_review > 0:
        try:
            query_embedding = embed_query(classified.cleaned)
        except Exception as e:
            logger.warning("Embedding failed, falling back to keyword-only: %s", e)

    if query_embedding and classified.weight_metadata > 0:
        try:
            ranked_lists.append((metadata_cosine_search(query_embedding, per_signal_limit), classified.weight_metadata))
        except Exception as e:
            logger.warning("Metadata cosine search failed: %s", e)

    if query_embedding and classified.weight_review > 0:
        try:
            ranked_lists.append((review_cosine_search(query_embedding, per_signal_limit), classified.weight_review))
        except Exception as e:
            logger.warning("Review cosine search failed: %s", e)

    if not ranked_lists:
        return []

    return reciprocal_rank_fusion(ranked_lists, k=settings.rrf_k, limit=limit)


def keyword_only_search(query: str, limit: int = 20) -> list[dict]:
    return bm25_search(query, limit=limit)


def semantic_only_search(query: str, limit: int = 20) -> list[dict]:
    try:
        query_embedding = embed_query(query)
    except Exception as e:
        logger.error("Embedding failed: %s", e)
        return []
    meta = metadata_cosine_search(query_embedding, limit=limit)
    review = review_cosine_search(query_embedding, limit=limit)
    return reciprocal_rank_fusion([(meta, 0.6), (review, 0.4)], k=settings.rrf_k, limit=limit)
