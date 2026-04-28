"""
Query classifier for the hybrid search pipeline.

Uses Claude Haiku to classify an incoming search query into one of four types,
then returns per-signal RRF weights (BM25 + metadata embedding + review embedding)
tuned for that query type.

Why Haiku: this runs on every search request, so latency and cost matter more
than raw reasoning power. Classification is a simple enough task for Haiku.

Why prompt caching: the system prompt is identical for every query. With caching,
repeated calls pay ~0.1x input cost instead of full price.

Usage:
    from lib.query_classifier import classify_query
    result = classify_query("dark atmospheric fantasy novels")
    # result.query_type     -> "thematic"
    # result.refined_query  -> "dark atmospheric fantasy novels"
    # result.bm25_weight    -> 0.1
    # result.metadata_weight -> 0.3
    # result.review_weight  -> 0.6
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import anthropic
from dotenv import load_dotenv

# Load .env so ANTHROPIC_API_KEY is available when running scripts directly.
load_dotenv(Path(__file__).parent.parent / ".env")

_client = anthropic.Anthropic()

# System prompt is frozen - never interpolate dynamic content here or caching breaks.
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

Respond only by calling the classify_query tool with all fields filled in.\
"""

# Tool schema - forces structured output via tool_choice.
_CLASSIFY_TOOL = {
    "name": "classify_query",
    "description": "Classify a search query and return RRF scoring weights.",
    "input_schema": {
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
    """Classify a search query and return RRF scoring weights.

    Makes one Claude Haiku API call. The system prompt is prompt-cached so
    repeated calls pay ~0.1x on the cached portion.

    Args:
        query: Raw user search string.

    Returns:
        QueryClassification with query_type, refined_query, and per-signal weights.

    Raises:
        anthropic.APIError: On API failure.
        ValueError: If the model returns an unexpected tool call name.
    """
    response = _client.messages.create(
        model="claude-haiku-4-5",
        max_tokens=256,
        system=[
            {
                "type": "text",
                "text": _SYSTEM_PROMPT,
                # Cache the system prompt - it never changes between requests.
                "cache_control": {"type": "ephemeral"},
            }
        ],
        tools=[_CLASSIFY_TOOL],
        # Force the model to always call classify_query - no free-text fallback.
        tool_choice={"type": "tool", "name": "classify_query"},
        messages=[{"role": "user", "content": query}],
    )

    # Extract the tool call result.
    tool_block = next(
        (b for b in response.content if b.type == "tool_use"),
        None,
    )
    if tool_block is None or tool_block.name != "classify_query":
        raise ValueError(f"Unexpected response from classifier: {response.content}")

    data = tool_block.input
    return QueryClassification(
        query_type=data["query_type"],
        refined_query=data["refined_query"],
        bm25_weight=float(data["bm25_weight"]),
        metadata_weight=float(data["metadata_weight"]),
        review_weight=float(data["review_weight"]),
        reasoning=data["reasoning"],
    )


if __name__ == "__main__":
    # Quick smoke test - run with: uv run python lib/query_classifier.py
    test_queries = [
        "Harry Potter and the Chamber of Secrets",
        "dark atmospheric fantasy with morally grey characters",
        "award-winning literary fiction 2023",
        "books like Gone Girl",
        "9780439023523",
    ]
    for q in test_queries:
        result = classify_query(q)
        print(f"\nQuery:    {q}")
        print(f"Type:     {result.query_type}")
        print(f"Refined:  {result.refined_query}")
        print(f"Weights:  BM25={result.bm25_weight}  metadata={result.metadata_weight}  review={result.review_weight}")
        print(f"Reason:   {result.reasoning}")
