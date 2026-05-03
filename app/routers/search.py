"""Search endpoints: /search, /search/keyword, /search/semantic"""
from __future__ import annotations
from fastapi import APIRouter, Query
from app.services.query_classifier import classify_query
from app.services.search import hybrid_search, keyword_only_search, semantic_only_search

router = APIRouter(prefix="/search", tags=["search"])


@router.get("")
def search(q: str = Query(..., min_length=1), limit: int = Query(default=20, ge=1, le=100)):
    """Hybrid search with automatic query classification (BM25 + vectors, RRF)."""
    classified = classify_query(q)
    results = hybrid_search(classified, limit=limit)
    return {
        "query": q,
        "query_type": classified.query_type,
        "weights": {
            "keyword": classified.weight_keyword,
            "metadata": classified.weight_metadata,
            "review": classified.weight_review,
        },
        "reasoning": classified.reasoning,
        "total": len(results),
        "results": results,
    }


@router.get("/keyword")
def search_keyword(q: str = Query(..., min_length=1), limit: int = Query(default=20, ge=1, le=100)):
    """BM25 / full-text search over ``books.search_tsv`` (see backfill script in repo)."""
    results = keyword_only_search(q, limit=limit)
    return {"query": q, "query_type": "keyword_only", "total": len(results), "results": results}


@router.get("/semantic")
def search_semantic(q: str = Query(..., min_length=1), limit: int = Query(default=20, ge=1, le=100)):
    """Vector-only semantic search."""
    results = semantic_only_search(q, limit=limit)
    return {"query": q, "query_type": "semantic_only", "total": len(results), "results": results}
