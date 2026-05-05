"""
FastAPI app — Library RAG Search & Recommendations.

Run with: uvicorn app.main:app --reload --port 8000
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
from lib.db import get_connection
from app.routers import analytics, books, inventory, search

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).parent.parent / "static"
API_VERSION = "1.2.0"


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
    if settings.groq_api_key:
        logger.info("Groq API key found — RAG recommendations enabled.")
    else:
        logger.info("No Groq key — RAG will use fallback mode.")
    yield
    logger.info("Shutdown complete.")


app = FastAPI(
    title="Library RAG — Search & Recommendations",
    description=(
        "Hybrid search: BM25 over books.search_tsv (GIN index), metadata and review vectors "
        "(pgvector), fused with RRF. RAG recommendations via Groq. Inventory and analytics included."
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
    db_ok = getattr(request.app.state, "database_configured", False)
    return {
        "service": "Library RAG API",
        "status": "running" if db_ok else "degraded",
        "version": API_VERSION,
        "database_configured": bool(db_ok),
        "docs": "/docs",
    }


if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
