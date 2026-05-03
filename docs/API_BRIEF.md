# Library RAG App — API & Search Brief

**Last updated:** April 30, 2026
**Written by:** Jahnavi Ravipudi

---

## What This Is

The FastAPI backend for the Intelligent Library Management and Recommendation System. It sits on top of the data ingestion pipeline and embedding generation layer, and provides the search, recommendation, and library management APIs that the frontend consumes.

```
┌──────────────────────────────────────────────────────────┐
│  Data Pipeline               Embeddings                  │
│  (962K books, 11M reviews)   (37K books with vectors)    │
│                                                          │
│              Neon Postgres + pgvector                     │
└────────────────────┬─────────────────────────────────────┘
                     │
          ┌──────────▼──────────┐
          │   app/ (this code)  │
          │                     │
          │  Query Classifier   │
          │  Hybrid Search      │
          │  RAG Pipeline       │
          │  Inventory Mgmt     │
          │  Analytics          │
          └──────────┬──────────┘
                     │
              FastAPI REST API
              localhost:8000/docs
                     │
          ┌──────────▼──────────┐
          │     Frontend        │
          └─────────────────────┘
```

---

## How Search Works

Every search query goes through a three-stage pipeline: **classify → retrieve → fuse**.

### Stage 1: Query Classification

The classifier determines what kind of query the user typed and sets per-signal weights accordingly. It first calls Groq (`lib/query_classifier.py`, OpenAI-compatible client, model from `GROQ_CLASSIFIER_MODEL`) when `GROQ_API_KEY` is set. If the call fails or the key is missing, it falls back to a local heuristic classifier.

| Query Type | Example | BM25 | Metadata | Review |
|---|---|---|---|---|
| **exact** | "Harry Potter", "9780439023523" | 0.80 | 0.10 | 0.10 |
| **thematic** | "dark atmospheric mystery", "cozy feel-good read" | 0.10 | 0.30 | 0.60 |
| **attribute** | "award-winning sci-fi 2020", "short stories under 200 pages" | 0.30 | 0.60 | 0.10 |
| **similarity** | "books like Gone Girl" | 0.10 | 0.45 | 0.45 |

The heuristic fallback uses regex patterns: ISBN detection, quoted title detection, "by Author" patterns, semantic indicator words (mood, vibe, theme adjectives), and title-case detection.

### Stage 2: Three Parallel Retrieval Signals

**Signal 1 — BM25 Keyword Search**
Uses Postgres full-text search (`to_tsvector` / `plainto_tsquery`) across title, authors, synopsis, genres, and subjects. Scores based on term frequency and proximity. Falls back to `ILIKE` pattern matching if full-text search returns nothing. Works on all 962K books. No embeddings needed.

**Signal 2 — Metadata Cosine Search**
Converts the user query into a 768-dim vector using BGE-base-en-v1.5, then finds the closest `books.metadata_embedding` vectors using pgvector's `<=>` cosine distance operator. Works on the 37K books that currently have embeddings. Captures what a book **is about** (publisher/catalog language).

**Signal 3 — Review Cosine Search**
Same vector approach but against `books.review_embedding`. Captures how readers **talk about** a book (mood, vibe, feeling). Bridges the vocabulary gap between catalog descriptions and natural user queries. Auto-detects whether review embeddings are pre-aggregated on the books table or need runtime aggregation from the reviews table.

### Stage 3: Reciprocal Rank Fusion (RRF)

The three ranked lists are merged using RRF with k=60 (Cormack et al., SIGIR'09). RRF operates on rank positions only — no score normalization needed between BM25's unbounded scores and cosine's [0,1] range.

```
RRF_score(book) = Σ  weight_signal / (k + rank_in_signal)
```

Books that appear high in multiple lists rise to the top. The weights from the classifier control how much each signal contributes.

### Graceful Degradation

If embeddings are missing (NULL), the vector signals return empty lists and RRF merges only the signals that returned results. The system degrades from three-signal hybrid search to keyword-only search without any code changes or errors.

---

## RAG Recommendation Pipeline

The `/recommend` endpoint adds a generation layer on top of search:

1. Classify the query
2. Run hybrid search → get top 8 books
3. Format book metadata (title, authors, genres, rating, synopsis) into a prompt
4. Send to Groq (Llama 3.3 70B) → get natural language recommendation
5. Return the recommendation text followed by the books

Without `GROQ_API_KEY`, falls back to a structured numbered list of the top 5 books with blurbs.

---

## API Reference

### Search Endpoints

| Method | Path | Description |
|---|---|---|
| GET | `/search?q=...&limit=20` | Hybrid search with auto-classification. Returns query_type, weights, reasoning, and ranked results. |
| GET | `/search/keyword?q=...&limit=20` | BM25-only search. No vector signals. Works on all 962K books. |
| GET | `/search/semantic?q=...&limit=20` | Vector-only search. Metadata + review cosine, fused with RRF. Only searches the 37K embedded books. |

**Response shape (all search endpoints):**
```json
{
  "query": "dark atmospheric mystery",
  "query_type": "thematic",
  "weights": {"keyword": 0.10, "metadata": 0.30, "review": 0.60},
  "reasoning": "Thematic/mood query",
  "total": 20,
  "results": [
    {
      "id": 12137,
      "isbn13": "9781409123781",
      "title": "Dark Matter",
      "authors": ["Michelle Paver"],
      "genres": ["Horror", "Fiction", "Mystery", "Gothic"],
      "synopsis": "January 1937. Clouds of war...",
      "goodreads_rating": 3.96,
      "num_ratings": 10762,
      "cover_image_url": "https://...",
      "rrf_score": 0.014442
    }
  ]
}
```

### Recommendation Endpoint

| Method | Path | Description |
|---|---|---|
| GET | `/recommend?q=...&limit=5` | RAG-powered recommendations. Returns books + natural language recommendation text. |

**Response shape:**
```json
{
  "query": "cozy fantasy with magic",
  "recommendation": "I'm excited to help you find your next cozy fantasy read with magic. I highly recommend Secondhand Spirits by Juliet Blackwell because it combines a cozy mystery with magical elements...",
  "query_type": "attribute",
  "books": [ ... ]
}
```

### Book Endpoints

| Method | Path | Description |
|---|---|---|
| GET | `/books/{isbn13}` | Full book details — all metadata fields, synopsis, plot_summary, cover image, awards. |
| GET | `/books/{isbn13}/reviews?limit=20&min_rating=3` | Reviews for a book. Filterable by minimum rating. |

### Inventory Endpoints

| Method | Path | Description |
|---|---|---|
| GET | `/availability/{isbn13}` | How many total and available copies exist for a book. |
| POST | `/borrow` | Borrow an available copy. Body: `{"isbn13": "...", "user_id": "..."}` |
| POST | `/return` | Return a borrowed book. Body: `{"borrow_id": 123}` |
| POST | `/renew` | Extend a borrow by 14 days (max 2 renewals). Body: `{"borrow_id": 123}` |

**Note:** Inventory is empty until copies are added to the `inventory` table. These endpoints are ready for the frontend but will return "No copies available" until then.

### Analytics Endpoints

| Method | Path | Description |
|---|---|---|
| GET | `/analytics/stats` | Database overview: total books, reviews, embedding counts, source breakdown. |
| GET | `/analytics/genres?limit=30` | Top genres by book count with average rating. |
| GET | `/analytics/popular?limit=20&period_days=30` | Most-borrowed books (empty until borrow activity exists). |

---

## How to Run

### Prerequisites
- Python 3.11+
- `.env` file with `DATABASE_URL_1` pointing to the Neon Postgres instance
- Groq API key (free) for RAG recommendations — get one at https://console.groq.com/keys

### Setup
```bash
# Clone the repo
git clone https://github.com/Devansh63/library-rag-ingest.git
cd library-rag-ingest

# Create .env from template and fill in values
copy .env.example .env
# Edit .env:
#   DATABASE_URL_1 = Neon connection string (remove &channel_binding=require if on Python 3.11)
#   GROQ_API_KEY = your Groq key (starts with gsk_)

# Start the server
uvicorn app.main:app --reload --port 8000

# Open Swagger docs
# http://localhost:8000/docs
```

### First startup
- Connects to Neon DB (may take 5-10 seconds if the DB is idle)
- Creates HNSW index on metadata_embedding for vector search (if pgvector is installed)
- Creates inventory, borrows, users tables (if not exist)
- Embedding model (BGE-base-en-v1.5, ~440MB) downloads on first search request

### Quick test
```bash
# Health check
curl http://localhost:8000/

# DB stats
curl http://localhost:8000/analytics/stats

# Keyword search
curl "http://localhost:8000/search/keyword?q=Harry+Potter"

# Hybrid search
curl "http://localhost:8000/search?q=dark+atmospheric+mystery"

# Semantic search
curl "http://localhost:8000/search/semantic?q=books+about+loneliness"

# RAG recommendation
curl "http://localhost:8000/recommend?q=cozy+fantasy+with+magic"
```

---

## What Happens on Startup

```
INFO:app.main:Testing database connection...        ← connects via lib.db.get_connection()
INFO:app.main:Database OK.
WARNING:app.main:Schema setup: functions in index    ← GIN index skip (harmless on Neon)
  expression must be marked IMMUTABLE
INFO:app.main:Search infrastructure ready.
INFO:app.main:Groq API key found — RAG              ← or "No Groq key — fallback mode"
  recommendations enabled.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://127.0.0.1:8000
```