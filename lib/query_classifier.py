"""
Groq-based query classifier for the hybrid search pipeline.

Classifies a query into one of four types and returns per-signal RRF weights
(BM25, metadata embedding, review embedding). Uses llama-3.1-8b-instant via
Groq's free tier - fast enough (~200ms) to run on every search request.

Usage:
    from lib.query_classifier import classify_query
    result = classify_query("dark atmospheric fantasy novels")
    # result.query_type    -> "thematic"
    # result.bm25_weight   -> 0.1
    # result.review_weight -> 0.6
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(Path(__file__).parent.parent / ".env")

_groq_client_singleton: OpenAI | None = None
_groq_client_sig: tuple[str, str, str] | None = None


def _groq_base_url() -> str:
    return (os.environ.get("GROQ_BASE_URL") or "https://api.groq.com/openai/v1").rstrip("/")


def _classifier_model() -> str:
    return os.environ.get("GROQ_CLASSIFIER_MODEL") or "llama-3.1-8b-instant"


def _groq_client() -> OpenAI:
    """Rebuild client when env vars change (e.g. key rotation in prod)."""
    global _groq_client_singleton, _groq_client_sig
    sig = (os.environ.get("GROQ_API_KEY", ""), _groq_base_url(), _classifier_model())
    if _groq_client_singleton is None or _groq_client_sig != sig:
        _groq_client_singleton = OpenAI(base_url=sig[1], api_key=sig[0], timeout=60.0)
        _groq_client_sig = sig
    return _groq_client_singleton


_SYSTEM_PROMPT = """\
You are a query classifier for a library book search and recommendation system.

The system fuses three search signals via Reciprocal Rank Fusion (RRF):
  - BM25: keyword matching (best for exact titles, authors, ISBNs, quoted phrases)
  - Metadata embedding: semantic search over title + genres + synopsis
    (best for genre/topic/attribute queries in formal language)
  - Review embedding: semantic search over reader-written reviews
    (best for mood/vibe/feeling queries in casual reader language)

Classify the query into one of four types and return weights that sum to 1.0:

  exact      - Known title, author name, ISBN, or quoted phrase
               bm25=0.80  metadata=0.10  review=0.10

  thematic   - Mood, vibe, or genre described in reader language
               ("dark atmospheric fantasy", "cozy mystery with a cat", "feel-good summer read")
               bm25=0.10  metadata=0.30  review=0.60

  attribute  - Structured properties: awards, page count, year, genre category
               ("award-winning sci-fi 2020", "classic Russian literature", "short stories under 200 pages")
               bm25=0.30  metadata=0.60  review=0.10

  similarity - "books like X" or "if I liked X what should I read"
               bm25=0.10  metadata=0.45  review=0.45

Also return a refined_query: clean up typos, expand obvious abbreviations,
and strip filler words. Keep the meaning intact - do not add words the user
did not imply.

You must respond by calling the classify_query function.\
"""

_CLASSIFY_TOOL = {
    "type": "function",
    "function": {
        "name": "classify_query",
        "description": "Classify a search query and return RRF scoring weights.",
        "parameters": {
            "type": "object",
            "properties": {
                "query_type": {
                    "type": "string",
                    "enum": ["exact", "thematic", "attribute", "similarity"],
                    "description": "Query category.",
                },
                "refined_query": {
                    "type": "string",
                    "description": "Cleaned/expanded version of the query for retrieval.",
                },
                "bm25_weight": {
                    "type": "number",
                    "description": "Weight for BM25 keyword signal (0.0-1.0).",
                },
                "metadata_weight": {
                    "type": "number",
                    "description": "Weight for metadata embedding signal (0.0-1.0).",
                },
                "review_weight": {
                    "type": "number",
                    "description": "Weight for review embedding signal (0.0-1.0).",
                },
                "reasoning": {
                    "type": "string",
                    "description": "One sentence explaining the classification decision.",
                },
            },
            "required": [
                "query_type",
                "refined_query",
                "bm25_weight",
                "metadata_weight",
                "review_weight",
                "reasoning",
            ],
        },
    },
}


@dataclass(frozen=True)
class QueryClassification:
    query_type: str
    refined_query: str
    bm25_weight: float
    metadata_weight: float
    review_weight: float
    reasoning: str


def classify_query(query: str) -> QueryClassification:
    """Classify a query and return RRF weights. Calls Groq once (free tier).

    Raises:
        openai.APIError: on Groq API failure.
        ValueError: if the model returns no tool call.
    """
    response = _groq_client().chat.completions.create(
        model=_classifier_model(),
        max_tokens=256,
        temperature=0.0,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": query},
        ],
        tools=[_CLASSIFY_TOOL],
        tool_choice={"type": "function", "function": {"name": "classify_query"}},
    )

    tool_calls = response.choices[0].message.tool_calls
    if not tool_calls or tool_calls[0].function.name != "classify_query":
        raise ValueError(f"Unexpected response from classifier: {response.choices[0].message}")

    data = json.loads(tool_calls[0].function.arguments)
    return QueryClassification(
        query_type=data["query_type"],
        refined_query=data["refined_query"],
        bm25_weight=float(data["bm25_weight"]),
        metadata_weight=float(data["metadata_weight"]),
        review_weight=float(data["review_weight"]),
        reasoning=data["reasoning"],
    )


if __name__ == "__main__":
    # Quick smoke test: uv run python lib/query_classifier.py
    test_queries = [
        "Harry Potter and the Chamber of Secrets",
        "dark atmospheric fantasy with morally grey characters",
        "award-winning literary fiction 2023",
        "books like Gone Girl",
        "9780439023523",
    ]
    for q in test_queries:
        result = classify_query(q)
        print(f"\nQuery:   {q}")
        print(f"Type:    {result.query_type}")
        print(f"Refined: {result.refined_query}")
        print(f"Weights: BM25={result.bm25_weight}  metadata={result.metadata_weight}  review={result.review_weight}")
        print(f"Reason:  {result.reasoning}")
