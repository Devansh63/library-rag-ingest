from __future__ import annotations

import os
from dataclasses import dataclass, field
from dotenv import load_dotenv

load_dotenv()


@dataclass
class Settings:
    # Embedding model - must match what was used to generate stored vectors.
    embedding_model: str = "BAAI/bge-base-en-v1.5"
    embedding_dim: int = 768

    # Search defaults
    default_search_limit: int = 20
    rrf_k: int = 60
    weight_metadata: float = 0.4
    weight_review: float = 0.3
    weight_keyword: float = 0.3

    # Groq (OpenAI-compatible API)
    groq_api_key: str = field(default_factory=lambda: os.environ.get("GROQ_API_KEY", ""))
    groq_base_url: str = field(
        default_factory=lambda: (
            os.environ.get("GROQ_BASE_URL") or "https://api.groq.com/openai/v1"
        ).rstrip("/")
    )
    groq_classifier_model: str = field(
        default_factory=lambda: os.environ.get("GROQ_CLASSIFIER_MODEL") or "llama-3.1-8b-instant"
    )
    rag_model: str = field(
        default_factory=lambda: os.environ.get("GROQ_RAG_MODEL") or "llama-3.3-70b-versatile"
    )
    rag_top_k: int = 8

    # Inventory
    max_borrow_days: int = 14
    max_renewals: int = 2


settings = Settings()
