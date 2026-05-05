# Intelligent Library Management and Recommendation System

**CS 410 Group 3 — University of Illinois Urbana-Champaign**

A hybrid search and RAG-powered book recommendation system built on top of a 962K-book catalog with 11M reader reviews.

---

## Demo

[![Library RAG Demo](https://cdn.loom.com/sessions/thumbnails/ab5474e9d6a84b83bea93a62c1094842-with-play.gif)](https://www.loom.com/share/ab5474e9d6a84b83bea93a62c1094842)

---

## What It Does

- **Hybrid search** — combines BM25 keyword search, metadata semantic search, and review semantic search via Reciprocal Rank Fusion (RRF)
- **Query classification** — automatically detects whether a query is exact (title/ISBN), thematic (mood/vibe), attribute (genre/awards), or similarity-based ("books like X"), then weights the three signals accordingly
- **Conversational RAG** — multi-turn librarian chat powered by Groq (llama-3.3-70b-versatile), grounded in catalog excerpts so it never hallucinates books
- **Library management** — inventory, borrowing, renewals, and analytics endpoints
- **Live on Render** at [library-rag.onrender.com](https://library-rag.onrender.com)

---

## Architecture

```
Data Sources                  Pipeline
------------                  --------
UCSD Book Graph  --+
Goodreads BBE    --+--> ingest -> ISBNdb enrichment -> embedding
CMU Summaries    --+         (962K books, 11M reviews)

                    Neon Postgres + pgvector
                    58K books with 768-dim vectors

                         FastAPI Backend
                    +--------------------+
                    |  Query Classifier  |  Groq llama-3.1-8b
                    |  BM25 Search       |  Postgres FTS
                    |  Metadata Search   |  pgvector cosine
                    |  Review Search     |  pgvector cosine
                    |  RRF Fusion        |
                    |  RAG Pipeline      |  Groq llama-3.3-70b
                    +--------------------+
                         React Frontend
```

---

## How Search Works

Every query goes through three stages: **classify -> retrieve -> fuse**

### Query Classification

| Type | Example | BM25 | Metadata | Review |
|------|---------|------|----------|--------|
| exact | "Harry Potter", ISBN | 0.80 | 0.10 | 0.10 |
| thematic | "dark atmospheric mystery" | 0.10 | 0.30 | 0.60 |
| attribute | "award-winning sci-fi 2020" | 0.30 | 0.60 | 0.10 |
| similarity | "books like Gone Girl" | 0.10 | 0.45 | 0.45 |

### Three Retrieval Signals

**BM25** — Postgres full-text search over title, authors, synopsis, and genres. Works on all 962K books. Falls back to ILIKE pattern matching when FTS returns nothing.

**Metadata embedding** — BGE-base-en-v1.5 (768-dim) vectors built from `Genres: {genres}. {title}. {synopsis}`. Captures what a book IS in publisher language.

**Review embedding** — BGE vectors built from KMeans-clustered reader reviews. Picks the most representative review from each of 5 perspective clusters (medoid selection), concatenates them, and embeds the result. Captures what readers SAY about the book — bridges the gap between how books are described vs how users search.

### RRF Fusion

Results from all three signals are merged with Reciprocal Rank Fusion: `score = weight / (k + rank)`, summed across signals. Default `k=60`.

---

## Embedding Pipeline

Books go through a multi-phase embedding pipeline:

1. **ISBNdb enrichment** (`scripts/enrich_isbndb.py`) — adds authors, publisher, and description to raw UCSD Graph books using the ISBNdb REST API. Books are flagged `CLEANED` once enriched.
2. **Quality scoring** (`scripts/embedding/mark_embed_queue.py`) — scores books by source quality, review count, and description richness. Marks the top candidates `EMBED_QUEUED`.
3. **Embedding** (`scripts/embedding/embed_sample.py`) — runs BGE-base-en-v1.5 locally on Apple Silicon (MPS) in 1,000-book batches. Writes `metadata_embedding` and `review_embedding` to Neon Postgres.

---

## Stack

| Layer | Technology |
|-------|------------|
| Database | Neon serverless Postgres + pgvector |
| Embeddings (ingest) | BAAI/bge-base-en-v1.5 via sentence-transformers (MPS) |
| Embeddings (runtime) | HuggingFace Inference API (no local model on Render) |
| Query classifier | Groq llama-3.1-8b-instant |
| RAG | Groq llama-3.3-70b-versatile |
| Backend | FastAPI + uvicorn |
| Frontend | React 18 (CDN, single HTML file) |
| Hosting | Render free tier |

---

## Local Setup

```bash
git clone https://github.com/Devansh63/library-rag-ingest
cd library-rag-ingest

pip install -r requirements.txt

cp .env.example .env
# Fill in DATABASE_URL_1, GROQ_API_KEY, HF_TOKEN

uvicorn app.main:app --reload --port 8000
# App at http://localhost:8000
# API docs at http://localhost:8000/docs
```

### Environment Variables

| Variable | Required | Purpose |
|----------|----------|---------|
| `DATABASE_URL_1` | Yes | Neon Postgres connection string |
| `GROQ_API_KEY` | Yes | Query classification + RAG |
| `HF_TOKEN` | Yes | Query embedding via HF Inference API |
| `ISBNDB_API_KEY` | Ingest only | Book metadata enrichment |

---

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/search?q=...&mode=hybrid` | Hybrid / keyword / semantic search |
| GET | `/books/{isbn13}` | Book detail + reviews |
| POST | `/recommend` | Single-shot RAG recommendation |
| POST | `/recommend/chat` | Multi-turn librarian chat |
| GET | `/analytics/stats` | Catalog statistics |
| GET | `/analytics/genres` | Genre distribution |
| GET | `/inventory/{isbn13}` | Availability check |
| GET | `/health` | Service liveness |

Full interactive docs at `/docs`.

---

## Project Structure

```
app/
  core/          config, db helpers
  routers/       search, books, inventory, analytics
  services/      search (BM25 + RRF), rag, query_classifier, inventory
lib/
  db.py          Neon connection
  query_classifier.py  Groq-based classifier
scripts/
  ingest_*.py    Data ingestion (UCSD, Goodreads, CMU)
  enrich_isbndb.py     ISBNdb enrichment
  embedding/
    mark_embed_queue.py  Quality scoring + queue
    embed_sample.py      BGE embedding pipeline
  dedup_reviews.py
  backfill_search_tsv.py
static/
  index.html     Single-file React frontend
```

---

## Team

**CS 410 Group 3 — Spring 2026**

- Devansh Agrawal
- Jahnavi Ravipudi
- Sahil (sahil211999)
- Omer Sajid
