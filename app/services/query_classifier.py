"""
Query Classifier — wraps lib/query_classifier.py (Claude Haiku-based).

Devansh's classifier uses Claude Haiku to classify queries into 4 types:
    exact      — known title/author/ISBN         (BM25-heavy)
    thematic   — mood/vibe in reader language     (review-embedding-heavy)
    attribute  — structured properties/genres     (metadata-embedding-heavy)
    similarity — "books like X"                   (balanced embeddings)

Falls back to lightweight heuristics if ANTHROPIC_API_KEY is not set
or the API call fails.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ClassifiedQuery:
    """Unified result for both Claude-based and fallback classification."""
    original: str
    query_type: str
    cleaned: str
    weight_keyword: float
    weight_metadata: float
    weight_review: float
    reasoning: str = ""


def classify_query(query: str) -> ClassifiedQuery:
    """
    Classify using Devansh's Claude Haiku classifier.
    Falls back to heuristics if the API call fails.
    """
    try:
        from lib.query_classifier import classify_query as haiku_classify
        result = haiku_classify(query)
        return ClassifiedQuery(
            original=query,
            query_type=result.query_type,
            cleaned=result.refined_query,
            weight_keyword=result.bm25_weight,
            weight_metadata=result.metadata_weight,
            weight_review=result.review_weight,
            reasoning=result.reasoning,
        )
    except Exception as e:
        logger.warning("Claude classifier failed (%s), using heuristic fallback", e)
        return _heuristic_fallback(query)


# ---------------------------------------------------------------------------
# Heuristic fallback (no API needed)
# ---------------------------------------------------------------------------

_ISBN_PATTERN = re.compile(r"^[\d\-]{10,17}[xX]?$")
_QUOTED_PATTERN = re.compile(r"""["'].+["']""")
_BY_AUTHOR_PATTERN = re.compile(r"\bby\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+")
_TITLE_CASE_PATTERN = re.compile(
    r"^[A-Z][a-z]+(?:\s+(?:[A-Z][a-z]+|the|of|and|in|a|an|for|to|or))+$"
)

_SEMANTIC_INDICATORS = {
    "about", "like", "similar", "mood", "vibe", "feel", "theme",
    "atmospheric", "dark", "funny", "sad", "cozy", "intense",
    "emotional", "romantic", "recommend", "suggest", "something", "anything",
}

_SEMANTIC_PHRASES = re.compile(
    r"books?\s+(?:about|like|similar\s+to|that)"
    r"|(?:looking|searching)\s+for"
    r"|(?:recommend|suggest)\s+(?:me|some|a)"
    r"|something\s+(?:like|similar|about)"
    r"|(?:best|top|great|good)\s+(?:books?|novels?|reads?|thriller|mystery|sci-?fi|fantasy)",
    re.IGNORECASE,
)


def _heuristic_fallback(query: str) -> ClassifiedQuery:
    original = query.strip()
    cleaned = original

    if _ISBN_PATTERN.match(re.sub(r"[\s\-]", "", original)):
        return ClassifiedQuery(original=original, query_type="exact",
                               cleaned=re.sub(r"[\s\-]", "", original),
                               weight_keyword=0.80, weight_metadata=0.10, weight_review=0.10,
                               reasoning="ISBN detected")

    if _QUOTED_PATTERN.search(original):
        cleaned = re.sub(r"""["']""", "", original).strip()
        return ClassifiedQuery(original=original, query_type="exact", cleaned=cleaned,
                               weight_keyword=0.80, weight_metadata=0.10, weight_review=0.10,
                               reasoning="Quoted title")

    words = set(original.lower().split())
    semantic_score = 0
    exact_score = 0

    if _SEMANTIC_PHRASES.search(original):
        semantic_score += 3
    semantic_score += len(words & _SEMANTIC_INDICATORS)

    if _BY_AUTHOR_PATTERN.search(original):
        exact_score += 3
    if _TITLE_CASE_PATTERN.match(original) and len(words) <= 6:
        exact_score += 2
    if len(words) <= 3 and semantic_score == 0:
        exact_score += 1

    if semantic_score >= 2 and exact_score == 0:
        return ClassifiedQuery(original=original, query_type="thematic", cleaned=cleaned,
                               weight_keyword=0.10, weight_metadata=0.30, weight_review=0.60,
                               reasoning="Thematic/mood query")

    if exact_score >= 2 and semantic_score == 0:
        return ClassifiedQuery(original=original, query_type="exact", cleaned=cleaned,
                               weight_keyword=0.80, weight_metadata=0.10, weight_review=0.10,
                               reasoning="Exact title/author lookup")

    return ClassifiedQuery(original=original, query_type="attribute", cleaned=cleaned,
                           weight_keyword=0.30, weight_metadata=0.40, weight_review=0.30,
                           reasoning="Ambiguous — balanced weights")
