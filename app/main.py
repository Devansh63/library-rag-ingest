"""
FastAPI app — Library RAG Search & Recommendations.

Uses ``.env`` for ``DATABASE_URL_1``, ``GROQ_API_KEY`` (optional), etc.

Run with: ``uvicorn app.main:app --reload --port 8000``
"""
from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.core.config import settings
from app.core.db import execute_write
from lib.db import get_connection
from app.routers import analytics, books, inventory, search

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).parent.parent / "static"

# Bumped when search schema, routes, or response shapes change meaningfully.
API_VERSION = "1.2.0"


def _ensure_search_infrastructure():
    """Create search indexes and missing tables. Idempotent."""
    statements = [
        # BM25: stored tsvector + GIN (expression index on array_to_string fails — not IMMUTABLE).
        """ALTER TABLE books ADD COLUMN IF NOT EXISTS search_tsv tsvector""",
        """CREATE INDEX IF NOT EXISTS idx_books_search_tsv ON books USING gin (search_tsv)
            WHERE search_tsv IS NOT NULL""",
        """DROP INDEX IF EXISTS idx_books_fts""",
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
    if not os.environ.get("DATABASE_URL_1"):
        logger.warning(
            "DATABASE_URL_1 is not set — copy .env.example to .env and add your Neon URL. "
            "The UI at / still loads; catalog and search API routes will fail until the database is configured."
        )
        app.state.database_configured = False
        if settings.groq_api_key:
            logger.info("Groq API key found — RAG will work once DATABASE_URL_1 is set.")
        else:
            logger.info("No Groq key — RAG will use fallback mode when the DB is configured.")
        yield
        logger.info("Shutdown complete.")
        return

    app.state.database_configured = True
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
    description=(
        "Hybrid search: BM25 over ``books.search_tsv`` (GIN; backfill via "
        "``scripts/backfill_search_tsv.py``), metadata and review vectors "
        "(pgvector), fused with RRF. RAG recommendations via Groq when "
        "``GROQ_API_KEY`` is set. Inventory and analytics endpoints included."
    ),
    version=API_VERSION,
    lifespan=lifespan,
    openapi_tags=[
        {"name": "search", "description": "Hybrid, keyword-only, and semantic search."},
        {"name": "books", "description": "Book detail, reviews, and RAG recommendations."},
        {"name": "inventory", "description": "Availability, borrow, return, renew."},
        {"name": "analytics", "description": "Stats, genres, popular borrows."},
        {"name": "health", "description": "Service liveness and configuration flags."},
    ],
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
    """Serve the frontend UI at root, or return JSON health check."""
    index = STATIC_DIR / "index.html"
    if index.exists():
        return FileResponse(index)
    return {
        "service": "Library RAG API",
        "status": "running",
        "version": API_VERSION,
        "docs": "/docs",
    }


@app.get("/health", tags=["health"])
def health(request: Request):
    """Liveness plus whether DATABASE_URL_1 was present at startup."""
    db_ok = getattr(request.app.state, "database_configured", False)
    return {
        "service": "Library RAG API",
        "status": "running" if db_ok else "degraded",
        "version": API_VERSION,
        "database_configured": bool(db_ok),
        "docs": "/docs",
    }


# Mount static files last so API routes take priority
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
