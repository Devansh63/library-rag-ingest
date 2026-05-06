# Intelligent Library Management and Recommendation System

**CS 410 — University of Illinois Urbana-Champaign**

Akash Muthukumar · Bhavana Thumma · Devansh Agrawal · Jahnavi Ravipudi · Sahil Sashi

A hybrid search and RAG-powered book recommendation system built on 962K books and 11M reader reviews.

---

## Demo

[![Library RAG Demo](https://cdn.loom.com/sessions/thumbnails/ab5474e9d6a84b83bea93a62c1094842-with-play.gif)](https://www.loom.com/share/ab5474e9d6a84b83bea93a62c1094842)

**Live deployment:** [library-rag-ingest.onrender.com](https://library-rag-ingest.onrender.com)

---

## Motivation

Traditional library catalogs handle exact queries well ("Harry Potter", an ISBN) but fail on thematic queries: "a cozy fantasy with magic," "something dark and atmospheric set in the Arctic with that Gone Girl feeling." Three failure modes cause this:

- **No semantic understanding** - a search for "cozy mystery" fails if the synopsis never uses the word *cozy*, even when every reader review does
- **Vocabulary mismatch** - publishers describe books in catalog language; readers describe them in emotional, thematic terms
- **No cross-signal ranking** - keyword match and semantic similarity each produce their own ranked list with no principled way to combine them

This system closes all three gaps by combining BM25 keyword retrieval, dense semantic search over catalog metadata, and dense semantic search over reader reviews, merged with Reciprocal Rank Fusion. A RAG layer converts retrieved candidates into grounded, natural-language recommendations.

---

## Architecture

```
Data Sources                  Pipeline
------------                  --------
Goodreads BBE    --+
CMU Summaries    --+--> ingest -> ISBNdb enrichment -> embedding
UCSD Book Graph  --+         (962K books, 11M reviews)

                    Neon Postgres + pgvector
                    58K books with metadata embeddings
                    56K books with review embeddings

                         FastAPI Backend
                    +-------------------------------+
                    |  Query Classifier             |  Groq llama-3.1-8b
                    |  BM25 (GIN index, 962K books) |  Postgres FTS
                    |  Metadata cosine (HNSW)       |  pgvector
                    |  Review cosine (HNSW)         |  pgvector
                    |  RRF Fusion  k=60             |
                    |  RAG generation               |  Groq llama-3.3-70b
                    +-------------------------------+
                         React SPA (static/index.html)
```

---

## How Search Works

Every query goes through three stages: **classify -> retrieve -> fuse**

### Stage 1 - Query Classification

A Groq-hosted `llama-3.1-8b-instant` classifier receives the query as a forced tool call and returns a structured JSON intent label plus per-signal weights. A regex-based fallback classifier runs when no API key is configured.

| Query type | Example | BM25 | Metadata | Review |
|------------|---------|------|----------|--------|
| exact | "Harry Potter", ISBN | 0.80 | 0.10 | 0.10 |
| thematic | "dark atmospheric mystery" | 0.10 | 0.30 | 0.60 |
| attribute | "award-winning sci-fi 2020" | 0.30 | 0.60 | 0.10 |
| similarity | "books like Gone Girl" | 0.10 | 0.45 | 0.45 |

### Stage 2 - Three Parallel Retrieval Signals

**BM25** - Postgres full-text search via a GIN index on a precomputed `search_tsv` column covering title, authors, synopsis, genres, and subjects. Works on all 962K books. Falls back to `ILIKE` pattern matching when FTS returns nothing.

**Metadata embedding** - BGE-base-en-v1.5 (768-dim) vectors built from `Genres: {genres}. {title}. {synopsis}`. Captures what a book *is* in publisher/catalog language.

**Review embedding** - BGE vectors built from KMeans-clustered reader reviews. Captures what readers *say* about a book. See the embedding pipeline section below.

### Stage 3 - Reciprocal Rank Fusion

The three ranked lists are merged with RRF (k=60):

```
RRF(d) = w1/(k + r1(d)) + w2/(k + r2(d)) + w3/(k + r3(d))
```

RRF operates on rank positions only, so no normalization is needed between BM25's unbounded scores and cosine's [0,1] values. A book ranked high across two or three signals reliably outranks a book that dominates only one. When embeddings are absent, the vector signals return empty lists and the search gracefully degrades to keyword-only.

---

## Review Embedding Pipeline

Each book's `review_embedding` is built via a multi-step procedure designed to capture distinct reader perspectives rather than averaging them into a single point:

```
Up to 60 reader reviews
        |
Embed each review individually (BGE-base-en-v1.5 -> 768-dim)
        |
KMeans clustering (k=5, random_state=42)
        |
Select medoid per cluster
(real review closest to centroid - not a synthetic average)
        |
Concatenate 5 medoid texts (truncated to 80 words each, ~400 tokens)
        |
Re-embed concatenated text -> books.review_embedding (768-dim)
```

**Why medoids, not centroids?** A centroid is a synthetic average with no corresponding real text. A medoid is an actual reader quote - interpretable and traceable. The 5 clusters capture distinct reader perspectives: enthusiastic fan, critical reader, casual reader, thematic reviewer, plot-focused reviewer.

---

## Data Pipeline

| Source | Books | Role |
|--------|-------|------|
| Goodreads BBE | 52,605 | ISBN-anchored core catalog |
| CMU Book Summaries | 11,893 | Plot summaries (matched by fuzzy title-author key, no ISBNs) |
| UCSD Book Graph | 893,453 | Ratings, reviews, cover images (privacy-scrubbed, no author names) |
| ISBNdb API | ongoing | Daily enrichment - adds authors, publisher, description (5K calls/day) |

Each source fills NULL fields on rows already laid down by earlier passes - no row is ever overwritten. The UCSD source uses a dual foreign-key strategy for reviews: linked by `isbn13` for the 651K books that have one, and by integer `book_id` for the 288K that do not.

### Current Data State

| Metric | Count |
|--------|-------|
| Books, total | 961,951 |
| With ISBN-13 | 651,128 |
| With author data | 69,780 |
| With synopsis | 814,042 |
| With metadata embedding | 58,748 |
| With review embedding | 56,439 |
| Reviews, total | 11,048,036 |

---

## Database Schema

Two main tables hold the catalog. `books` stores one row per book including a `vector(768)` metadata embedding. `reviews` stores one row per review including a `vector(768)` review embedding. Three smaller tables (`inventory`, `borrows`, `users`) are created lazily at startup for the library-management endpoints.

The `books.cleaning_flags` array column tracks per-row data quality state used as a checkpoint for the enrichment pipeline - `isbndb_checked` marks rows already attempted, allowing safe resumption of partially completed runs.

Dense retrieval uses an HNSW index (`m=16`, `ef_construction=200`) over `metadata_embedding`. BM25 uses a GIN index over `search_tsv`. Both indexes are built offline; the FastAPI service performs no index creation at startup.

---

## RAG Recommendation

The `/recommend` endpoint adds a generation layer on top of search:

1. Classify the query and obtain per-signal weights
2. Run three-signal hybrid retrieval and fuse via RRF - select top 8 candidates
3. Format each book into a structured block: title, authors, genres, rating, synopsis excerpt
4. Send query + 8 book blocks to `llama-3.3-70b-versatile` (Groq, `max_tokens=800`, `temperature=0.7`) with a librarian persona prompt
5. Return both the structured book list and the generated prose

The LLM is never asked to recall books from memory - it only explains books the retrieval layer already selected. This prevents hallucinated titles. Without a Groq key, the endpoint returns a deterministic ranked list with synopsis blurbs.

The `/recommend/chat` endpoint extends this to a multi-turn conversational interface with fresh hybrid retrieval on each user turn.

---

## Key Design Decisions

**One database, no separate vector store** - BM25 and cosine search both run from the same Neon Postgres instance (GIN + HNSW). Avoids a dedicated vector database (Pinecone, Weaviate), keeps the operational footprint to a single managed dependency, and lets each signal be a single SQL query.

**RRF over score normalization** - BM25 produces unbounded scores; cosine produces values on [0,1]. Fusing by rank is invariant to score scale and well-attested in the IR literature.

**Hosted inference over local models** - The deployed image bundles no PyTorch or sentence-transformers. Query embedding is delegated to the HuggingFace Inference API; generation to Groq. This keeps the service compatible with Render's free tier (512MB RAM).

**Quality state on the row** - Cleaning state lives in a Postgres array column (`cleaning_flags`) rather than a separate table. Supports cheap GIN indexing and lets long-running enrichment jobs be killed and resumed without external checkpoints.

**English-only filter** - BGE-base-en-v1.5 is English-only. Non-English text produces vectors that do not cluster meaningfully. The pipeline hard-rejects non-English books.

**Graceful degradation** - Without a Groq key: classifier falls back to regex heuristics, recommendations return structured blurbs. Without an HF token: dense signals return empty, search falls back to BM25. Without ISBNdb credentials: enrichment is a no-op. Every external dependency has a fallback.

---

## Stack

| Layer | Technology |
|-------|------------|
| Database | Neon serverless Postgres + pgvector |
| Embeddings (ingest) | BAAI/bge-base-en-v1.5 via sentence-transformers (MPS/CPU) |
| Embeddings (runtime) | HuggingFace Inference API |
| Query classifier | Groq llama-3.1-8b-instant |
| RAG generation | Groq llama-3.3-70b-versatile |
| Backend | FastAPI + uvicorn |
| Frontend | React 18 (CDN, single HTML file) |
| Hosting | Render free tier |

---

## Local Setup

```bash
git clone https://github.com/Devansh63/intelligent-book-recommendation-system
cd intelligent-book-recommendation-system

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
| `GROQ_API_KEY` | Yes | Query classification + RAG (free tier) |
| `HF_TOKEN` | Yes | Query embedding via HF Inference API (free tier) |
| `ISBNDB_API_KEY` | Ingest only | Book metadata enrichment (5K calls/day, basic plan) |

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
| GET | `/health` | Service liveness + configuration flags |

Full interactive docs at `/docs`.

---

## Project Structure

```
app/
  core/            config.py, db.py
  routers/         search.py, books.py, inventory.py, analytics.py
  services/        search.py (BM25+RRF), rag.py, query_classifier.py, inventory.py
lib/
  db.py            Neon connection management
  query_classifier.py  Groq-based classifier with regex fallback
scripts/
  ingest_goodreads_bbe.py
  ingest_cmu_summaries.py
  ingest_ucsd_graph.py
  enrich_isbndb.py       ISBNdb daily enrichment job
  dedup_reviews.py
  backfill_search_tsv.py
  embedding/
    mark_embed_queue.py  Quality scoring + EMBED_QUEUED flag
    embed_sample.py      BGE embedding pipeline (MPS-accelerated)
static/
  index.html       Single-file React frontend
docs/
  API_BRIEF.md     Detailed API documentation
```

---

## Limitations and Future Work

**Current limitations:**
- Embedding coverage is ~6% of the full catalog (58K of 962K books). Dense signals cover the embedded subset; BM25 covers all 962K. Coverage grows with each daily ISBNdb + embedding run.
- English-only: no multilingual query support.
- ISBNdb enrichment is rate-limited to 5K calls/day, meaning full catalog enrichment takes ~29 more daily runs.
- No formal FK constraints on review join paths - integrity is enforced by ingestion scripts.

**Planned directions:**
- Extend embedding coverage to the full catalog
- Evaluation harness with curated relevance judgments for systematic A/B comparison of weight schedules
- Multilingual support via a multilingual embedding model
- Personalization using borrow history from the `borrows` table
- Inventory integration - filter/rank results by real-time availability

---

## Paper

[Final report (PDF)](https://github.com/Devansh63/intelligent-book-recommendation-system/blob/main/docs/CS410_final_report.pdf)

---

## Team

**CS 410 - Spring 2026 - University of Illinois Urbana-Champaign**

| Name | Email |
|------|-------|
| Akash Muthukumar | akashm4@illinois.edu |
| Bhavana Thumma | bthumma2@illinois.edu |
| Devansh Agrawal | dagraw2@illinois.edu |
| Jahnavi Ravipudi | jr84@illinois.edu |
| Sahil Sashi | ssashi2@illinois.edu |
