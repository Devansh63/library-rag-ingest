"""
FastAPI app — Library RAG Search & Recommendations.

Sits inside Devansh's library-rag-ingest repo alongside lib/ and scripts/.
Uses the same .env file (DATABASE_URL_1, ANTHROPIC_API_KEY).

Run with: uvicorn app.main:app --reload --port 8000
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import settings
from app.core.db import execute_write
from lib.db import get_connection
from app.routers import analytics, books, inventory, search

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _ensure_search_infrastructure():
    """Create search indexes and missing tables. Idempotent."""
    statements = [
        # GIN index for BM25 full-text search
        """CREATE INDEX IF NOT EXISTS idx_books_fts ON books USING gin(
            to_tsvector('english',
                coalesce(title, '') || ' ' ||
                coalesce(array_to_string(authors, ' '), '') || ' ' ||
                coalesce(synopsis, '') || ' ' ||
                coalesce(array_to_string(genres, ' '), '') || ' ' ||
                coalesce(array_to_string(subjects, ' '), '')
            )
        )""",
        # HNSW index for metadata vector search
        """DO $$ BEGIN
            IF EXISTS (SELECT 1 FROM pg_extension WHERE extname = 'vector') THEN
                IF NOT EXISTS (SELECT 1 FROM pg_indexes WHERE indexname = 'idx_books_metadata_embedding_hnsw') THEN
                    CREATE INDEX idx_books_metadata_embedding_hnsw
                    ON books USING hnsw (metadata_embedding vector_cosine_ops) WITH (m = 16, ef_construction = 200);
                END IF;
            END IF;
        END $$""",
        # inventory table
        """CREATE TABLE IF NOT EXISTS inventory (
            id SERIAL PRIMARY KEY, isbn13 VARCHAR(13),
            copy_number INTEGER DEFAULT 1, barcode VARCHAR(50),
            condition VARCHAR(20) DEFAULT 'good', location TEXT,
            date_acquired DATE DEFAULT CURRENT_DATE, is_active BOOLEAN DEFAULT true
        )""",
        # borrows table
        """CREATE TABLE IF NOT EXISTS borrows (
            id SERIAL PRIMARY KEY, inventory_id INTEGER REFERENCES inventory(id),
            user_id VARCHAR(100), borrow_date DATE DEFAULT CURRENT_DATE,
            due_date DATE NOT NULL, return_date DATE,
            status VARCHAR(20) DEFAULT 'active', renewed_count INTEGER DEFAULT 0
        )""",
        # users table
        """CREATE TABLE IF NOT EXISTS users (
            id SERIAL PRIMARY KEY, user_id VARCHAR(100) UNIQUE NOT NULL,
            name TEXT NOT NULL, email TEXT NOT NULL,
            join_date DATE DEFAULT CURRENT_DATE, favorite_genres TEXT[],
            is_active BOOLEAN DEFAULT true
        )""",
    ]
    for sql in statements:
        try:
            execute_write(sql)
        except Exception as e:
            logger.warning("Schema setup: %s", e)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Testing database connection...")
    conn = get_connection()
    conn.close()
    logger.info("Database OK.")
    _ensure_search_infrastructure()
    logger.info("Search infrastructure ready.")
    if settings.groq_api_key:
        logger.info("Groq API key found — RAG recommendations enabled.")
    else:
        logger.info("No Groq key — RAG will use fallback mode.")
    yield
    logger.info("Shutdown complete.")


app = FastAPI(
    title="Library RAG — Search & Recommendations",
    description="Hybrid search (BM25 + pgvector) with RAG recommendations.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(search.router)
app.include_router(books.router)
app.include_router(inventory.router)
app.include_router(analytics.router)


@app.get("/", tags=["health"])
def root():
    return {"service": "Library RAG API", "status": "running", "docs": "/docs"}
