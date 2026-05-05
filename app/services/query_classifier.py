"""
Query classifier wrapper around lib/query_classifier.py (Groq-based).

Four query types with corresponding RRF weights:
  exact     (BM25-heavy)          bm25=0.80  meta=0.10  review=0.10
  thematic  (review-embed-heavy)  bm25=0.10  meta=0.30  review=0.60
  attribute (meta-embed-heavy)    bm25=0.30  meta=0.60  review=0.10
  similarity (balanced embeds)    bm25=0.10  meta=0.45  review=0.45

Falls back to regex heuristics if GROQ_API_KEY is not set or the call fails.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ClassifiedQuery:
    original: str
    query_type: str
    cleaned: str
    weight_keyword: float
    weight_metadata: float
    weight_review: float
    reasoning: str = ""


def classify_query(query: str) -> ClassifiedQuery:
    try:
        from lib.query_classifier import classify_query as groq_classify
        result = groq_classify(query)
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
        logger.warning("Groq classifier failed (%s), using heuristic fallback", e)
        return _heuristic_fallback(query)


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
