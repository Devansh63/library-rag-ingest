"""
Application settings — reads from the same .env file.

Database connection: lib/db.get_connection() (reads DATABASE_URL_1)
Query classifier:   lib/query_classifier.py (reads GROQ_API_KEY)
Embeddings:         app/services/search.py  (reads HF_TOKEN)
RAG pipeline:       app/services/rag.py     (reads GROQ_API_KEY)
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from dotenv import load_dotenv

load_dotenv()


@dataclass
class Settings:
    # --- Embeddings (HuggingFace Inference API — no local model) ---
    embedding_model: str = "BAAI/bge-base-en-v1.5"
    embedding_dim: int = 768

    # --- Search defaults ---
    default_search_limit: int = 20
    rrf_k: int = 60
    weight_metadata: float = 0.4
    weight_review: float = 0.3
    weight_keyword: float = 0.3

    # --- RAG (uses Groq — free tier) ---
    groq_api_key: str = field(default_factory=lambda: os.environ.get("GROQ_API_KEY", ""))
    rag_model: str = "llama-3.3-70b-versatile"
    rag_top_k: int = 8

    # --- Inventory ---
    max_borrow_days: int = 14
    max_renewals: int = 2


settings = Settings()
