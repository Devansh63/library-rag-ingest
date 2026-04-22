# Library RAG Ingest - Project Brief

**Last updated:** April 21, 2026 (rev 2 - dual FK review linking)
**Written by:** Devansh Agrawal

---


## High-Level Architecture

```
Goodreads BBE (Zenodo)    --> books table (primary keys, ISBN-anchored)
CMU Book Summary Dataset  --> books table (adds plot_summary via title match)
UCSD Goodreads Book Graph --> books table (ratings, genres, cover images) + reviews table
ISBNdb API (ongoing)      --> books table (authors, publisher, language, subjects, etc.)

Embedding model           --> books.metadata_embedding, reviews.review_embedding
(BGE-base-en-v1.5)           [NOT YET RUN - teammate task]
```

The three dataset ingestors run in order. Each one either inserts new rows or fills NULL fields in rows already inserted by a prior source. ISBNdb runs daily as an ongoing enrichment job on top of that.

---

## Data Sources

### 1. Goodreads Best Books Ever (Zenodo) - `goodreads_bbe`
- ~52,000 books
- Highest-quality metadata: ISBNs, author names, genres, awards, Goodreads ratings
- Source of truth for ISBN keys. All later sources key off the isbn13 values this pass lays down.
- Script: `scripts/ingest_goodreads_bbe.py`

### 2. CMU Book Summary Dataset - `cmu_summaries`
- ~16,500 books total; ~11,900 ended up as new rows (rest matched Zenodo by title/author)
- Provides `plot_summary` field (distinct from `synopsis` - these are scholarly summaries, not publisher blurbs)
- No ISBN data, so matching is title+author fuzzy match
- Script: `scripts/ingest_cmu_summaries.py`

### 3. UCSD Goodreads Book Graph - `ucsd_graph`
- ~2.3M books raw; ~897,000 rows after quality filtering
- Provides: Goodreads ratings, user genres (popular shelves), cover images, publication dates
- Also the source of all 5.5M user reviews
- NOTE: The full UCSD dataset is privacy-scrubbed. Author names are removed (only `author_id` is present). Every UCSD-inserted book will have a `missing_author` flag until ISBNdb fills those in.
- Script: `scripts/ingest_ucsd_graph.py`

### 4. ISBNdb API (enrichment, not a primary source) - `isbndb`
- Daily enrichment job, not a full ingest
- Fills: author names, publisher, language, page count, subjects, ISBN-10, Dewey decimal
- Basic plan: 5,000 ISBN lookups per day, 100 per bulk API call
- Script: `scripts/enrich_isbndb.py`

---

## Database Schema

The database lives on Neon (serverless Postgres). Connection string is in `.env` as `DATABASE_URL_1`.

### books table

Every row is one book. The `source` column marks which dataset originally inserted the row.

| Column | Type | Description |
|---|---|---|
| `id` | serial | Auto-increment primary key |
| `isbn13` | varchar(13) | 13-digit ISBN. Nullable (CMU and some UCSD books lack ISBNs). Unique index used for dedup. |
| `isbn10` | varchar(10) | 10-digit ISBN. Filled by ISBNdb enrichment. |
| `title` | text | Full book title, HTML-unescaped |
| `authors` | text[] | List of author name strings. Empty array if unknown. |
| `publisher` | text | Publisher name |
| `publish_date` | date | First or primary publication date |
| `first_publish_date` | date | Earliest known publication date (if different from publish_date) |
| `genres` | text[] | Genre tags. From Goodreads shelves (UCSD) or Zenodo genre list. |
| `subjects` | text[] | Library subject headings. Filled by ISBNdb. Merged (not replaced) on enrichment. |
| `language` | varchar | ISO language code (e.g. "en", "eng"). NULL means unknown, DB defaults to 'en'. |
| `pages` | integer | Page count |
| `edition` | text | Edition string (e.g. "2nd Edition") |
| `series` | text | Series name if part of a series |
| `short_description` | text | Short blurb (typically 1-2 sentences from ISBNdb excerpt field) |
| `synopsis` | text | Full publisher description or back-cover text |
| `plot_summary` | text | Scholarly plot summary from CMU dataset |
| `cover_image_url` | text | URL to cover image |
| `goodreads_rating` | float | Average Goodreads rating (0.0 to 5.0) |
| `num_ratings` | integer | Number of Goodreads ratings |
| `awards` | text[] | Award names from Zenodo dataset |
| `source` | text | Which dataset inserted this row (see Source Values below) |
| `cleaning_flags` | text[] | Quality flags. NULL = not evaluated. Empty = evaluated and clean. Non-empty = has issues. (See Cleaning Flags below) |
| `ucsd_book_id` | text | UCSD's internal book_id, stored for all ucsd_graph rows. Used as the bridge to link reviews for books that have no isbn13. Only populated for ucsd_graph source rows. |
| `metadata_embedding` | vector(768) | BGE-base-en-v1.5 embedding of title + description + genres. NOT YET GENERATED. |
| `dewey_decimal` | text[] | Dewey Decimal classification codes from ISBNdb |

**Source values for `books.source`:**
- `goodreads_bbe` - Row was inserted by the Zenodo Goodreads Best Books Ever ingestor
- `cmu_summaries` - Row was inserted by the CMU Book Summary ingestor (no ISBN)
- `ucsd_graph` - Row was inserted by the UCSD Goodreads Graph ingestor
- `isbndb` - Reserved for future use; enrichment does not change the source column
- `merged` - Reserved for rows merged from 2+ sources (not yet implemented)

---

### reviews table

Every row is one review. Reviews are linked to books via one of two FK paths - `isbn13` for books that have one, `book_id` for books that do not. Exactly one of the two will be set per row.

| Column | Type | Description |
|---|---|---|
| `id` | serial | Auto-increment primary key |
| `isbn13` | varchar(13) | FK to books.isbn13. Set for reviews linked to isbn-bearing books. NULL for no-isbn books. |
| `book_id` | integer | FK to books.id. Set for reviews linked to no-isbn books where isbn13 is unavailable. NULL for isbn-bearing books. |
| `user_id` | text | Reviewer identifier. For UCSD reviews: the anonymized Goodreads user ID. For editorial reviews: the publication name (e.g. "Kirkus Reviews"). |
| `rating` | integer | Star rating, 1-5. NULL for editorial reviews (they have no star rating). |
| `review_text` | text | Full review text |
| `date_posted` | date | Date the review was posted |
| `spoiler_flag` | boolean | True if the original source flagged the review as containing spoilers |
| `source` | text | Which dataset the review came from (see Source Values below) |
| `review_type` | text | Either `user` or `editorial` (see Review Types below) |
| `review_embedding` | vector(768) | BGE-base-en-v1.5 embedding of review_text. NOT YET GENERATED. |

**Source values for `reviews.source`:**
- `ucsd_graph` - Review from the UCSD Goodreads Book Graph dataset
- `user_submitted` - Reserved for future user-submitted reviews via the application
- `isbndb` - Editorial review from the ISBNdb API

**Review type values for `reviews.review_type`:**
- `user` - Community review written by a Goodreads user. Has a star rating, has user_id.
- `editorial` - Professional review from a publication (Kirkus, Publishers Weekly, etc.). No star rating. `user_id` holds the publication name.

---

### Table Relationships

```
books (id, isbn13, ucsd_book_id)
  |
  +-- reviews (isbn13)  -- path 1: isbn-bearing books (~652K books)
  +-- reviews (book_id) -- path 2: no-isbn books (~288K books)
```

Reviews use two FK paths depending on whether the book has an isbn13:
- **Path 1** - `reviews.isbn13 = books.isbn13`: used by isbn-bearing books. All reviews from Zenodo/CMU sources and most UCSD books use this path.
- **Path 2** - `reviews.book_id = books.id`: used by no-isbn books. `books.ucsd_book_id` is the bridge - it stores UCSD's internal `book_id` so the ingestor can look up `books.id` after a bulk insert and link reviews correctly.

There is no formal FK constraint on either path (to allow flexibility during ingestion). To query reviews for any book regardless of path:
```sql
SELECT b.title, r.review_text
FROM books b
JOIN reviews r ON (b.isbn13 = r.isbn13 OR b.id = r.book_id)
WHERE b.title = 'Some Title';
```

The `metadata_embedding` and `review_embedding` columns are vector(768). Semantic search uses pgvector's `<=>` operator (cosine distance). These are NULL in all rows right now; a separate embedding pass needs to run.

---

## Schema Changes We Made (and Why)

### 1. `books.cleaning_flags` (text[])

**Why added:** The UCSD dataset has 2.3M books but wildly varying quality. We needed a way to mark books that passed the minimum bar for insertion (10+ ratings, English language, non-empty title) but still had fixable issues. Rather than using a separate table or file-based checkpoint, we embed the quality state directly in the row. This lets you query "what needs fixing" with a single `WHERE cleaning_flags @> ARRAY['missing_description']` clause and avoid re-processing books that have already been checked.

**NULL vs empty array vs populated:** NULL means the row was never evaluated (Zenodo and CMU rows are not evaluated because they are high-quality sources). An empty array `{}` means evaluated and clean. A non-empty array means there are specific known issues.

### 2. `reviews.review_type` (text, values: 'user' or 'editorial')

**Why added:** ISBNdb can return professional editorial reviews from publications like Kirkus Reviews. These are fundamentally different from user star ratings - they have no numeric rating, they are longer and more descriptive, and they come from a publication not a person. Without a separate column, queries for "user ratings" would mix in editorial content. Now: `WHERE review_type = 'user'` for community ratings, `WHERE review_type = 'editorial'` for professional reviews.

### 3. `books.ucsd_book_id` (text) and `reviews.book_id` (integer)

**Why added:** 288,080 books in the DB have no isbn13 (the UCSD dataset often omits ISBNs). Originally, reviews for these books were silently dropped during ingest because `reviews.isbn13` was the only way to link a review to a book. Since RAG quality depends on having reviews attached to as many books as possible, this was a significant gap.

The fix uses two new columns. `books.ucsd_book_id` stores UCSD's internal `book_id` for every ucsd_graph row. After the books ingest pass completes, a single query (`SELECT ucsd_book_id, id FROM books WHERE ucsd_book_id IS NOT NULL`) builds a complete map of UCSD id to `books.id`. The reviews pass then uses this map to set `reviews.book_id` for no-isbn books instead of `reviews.isbn13`.

A one-time backfill script (`scripts/backfill_ucsd_book_id.py`) was run to populate `ucsd_book_id` on all existing UCSD rows that were inserted before this column existed. After that, `--reviews-only` was re-run to recover all previously dropped reviews.

### 4. `books.dewey_decimal` (text[])

**Why added:** ISBNdb returns Dewey Decimal Classification codes (e.g. `["813.54", "FIC"]`). These are standard library classification codes and are directly useful for the library-specific search and browsing features we are building. The column is text[] not text because a book can have multiple Dewey codes.

---

## Cleaning Flags Reference

These flags appear in `books.cleaning_flags`. Multiple flags can be present on one row.

| Flag | Meaning | Fixable by |
|---|---|---|
| `missing_isbn` | No ISBN10 or ISBN13 anywhere in the source record | Manual lookup or title-based ISBNdb search |
| `missing_description` | No synopsis or description field at all | ISBNdb enrichment (priority 1) |
| `short_description` | Description exists but is under 50 characters | ISBNdb enrichment (partially) |
| `missing_author` | No author attribution (all UCSD books have this - source is privacy-scrubbed) | ISBNdb enrichment (priority 2) |
| `suspect_title` | Title is under 3 characters or purely numeric | Manual review |
| `low_rating_count` | Between 10 and 24 ratings (just above the minimum floor) | No fix needed - just a quality signal |
| `isbndb_checked` | ISBNdb API was queried for this book | - (checkpoint flag, not an issue) |
| `isbndb_not_found` | ISBN was not found in ISBNdb database | Try Open Library or Google Books |
| `CLEANED` | ISBNdb enrichment successfully filled at least one of: authors or synopsis | - (success marker) |

**Useful queries:**
```sql
-- Books still needing description (top priority for embedding quality):
SELECT COUNT(*) FROM books
WHERE cleaning_flags @> ARRAY['missing_description']
  AND NOT (cleaning_flags @> ARRAY['isbndb_checked']);

-- Books successfully enriched today:
SELECT COUNT(*) FROM books WHERE cleaning_flags @> ARRAY['CLEANED'];

-- Editorial reviews stored:
SELECT * FROM reviews WHERE review_type = 'editorial' LIMIT 10;
```

---

## Current Data Stats (as of April 21, 2026)

### books table
| Metric | Count |
|---|---|
| Total rows | 961,951 |
| From goodreads_bbe | 52,605 |
| From cmu_summaries | 11,893 |
| From ucsd_graph | 897,453 |
| With isbn13 | 652,589 |
| With author data | 69,780 |
| With synopsis | 808,046 |
| With metadata_embedding | 0 (not yet generated) |
| cleaning_flags NULL (not evaluated) | 64,498 |
| cleaning_flags set (UCSD rows) | 897,453 |
| CLEANED (ISBNdb enriched) | 7,669 |
| isbndb_checked | 7,675 |
| isbndb_not_found | 325 |
| Still need ISBNdb enrichment (missing_description, unchecked) | ~145,270 |

### reviews table
| Metric | Count |
|---|---|
| Total rows | 5,499,063 |
| From ucsd_graph | 5,499,063 |
| review_type = 'user' | 5,499,063 |
| review_type = 'editorial' | 0 (ISBNdb rarely returns these) |
| With review_embedding | 0 (not yet generated) |

### Flag breakdown (books)
| Flag | Count |
|---|---|
| missing_author | 889,789 |
| missing_isbn | 288,080 |
| low_rating_count | 276,466 |
| missing_description | 145,527 |
| isbndb_checked | 7,675 |
| CLEANED | 7,669 |
| short_description | 3,787 |
| isbndb_not_found | 325 |
| suspect_title | 228 |

**Note on missing_author:** Every UCSD row has this flag because the full UCSD dataset is privacy-scrubbed. Author names are not present - only internal `author_id` values. ISBNdb is the primary path to fill these. At 5,000 lookups/day, all 652K ISBN-bearing books can be enriched in about 130 days.

---

## What Still Needs to Be Done

### Your task - Embedding generation (not in this repo)

Both `books.metadata_embedding` and `reviews.review_embedding` are NULL for every row. These need to be generated using BGE-base-en-v1.5 (English-only, produces 768-dimension vectors).

Key details:
- Model: `BAAI/bge-base-en-v1.5` via HuggingFace Transformers
- Dimensions: 768 (matches the vector(768) column type)
- Input for `metadata_embedding`: concatenate `title + " " + synopsis + " " + array_to_string(genres, ' ')`
- Input for `review_embedding`: `review_text` directly
- Use batched inference (recommend batch size 64-128 on GPU, 16-32 on CPU)
- pgvector search operator: `<=>` for cosine distance, `<#>` for negative inner product

Skip rows where the input text is NULL or empty. Use `UPDATE books SET metadata_embedding = $1 WHERE id = $2` in batches using psycopg2 `execute_values`.

Connection: use `lib.db.get_connection()` from this repo - it reads `DATABASE_URL_1` from `.env`.

### Devansh's ongoing task - Daily ISBNdb enrichment

Run every day after 00:00 UTC:
```bash
uv run python scripts/enrich_isbndb.py --priority-only
```

This processes ~5,000 books per day (the daily API quota). At this rate, enriching all 145K missing-description books takes about 29 more days.

---

## How to Run (if you need to)

Prerequisites: Python 3.12+, `uv` package manager, `.env` file with `DATABASE_URL_1` and `ISBNDB_API_KEY`.

```bash
# Install deps
uv sync

# Run any script (uv handles the venv automatically)
uv run python scripts/enrich_isbndb.py --dry-run --limit 200
uv run python scripts/ingest_ucsd_graph.py --dry-run --limit 1000
```

---

## Key Design Decisions

**Why batch UPDATEs instead of per-row UPDATEs?**
The UCSD dataset has 2.3M books. If we matched even 10% against existing rows, that's 230K UPDATE statements. At 80ms per round-trip to Neon (serverless cold start latency), that's 5+ hours. Instead, we collect updates in a list and flush them as a single `UPDATE ... FROM VALUES` round trip per 100-row batch. This cuts the update pass from hours to minutes.

**Why cleaning_flags instead of a separate quality table?**
It keeps the quality state colocated with the row. No JOIN needed to check status. Array operators (`@>`) are fast with a GIN index. And DB-based checkpointing means you can kill the enrichment job mid-run and resume - it picks up from wherever it stopped.

**Why hard-reject non-English books?**
The embedding model (BGE-base-en-v1.5) is English-only. Non-English text through this model produces vectors that do not cluster meaningfully - they would pollute nearest-neighbor results. Books with no language code are accepted (the UCSD dataset is heavily English-biased and missing language_code is common).

**Why cap reviews at 60 per book?**
The review embedding for a book is computed by averaging its reviews. Beyond about 60 reviews, the averaged vector stabilizes and additional reviews add noise rather than signal. Capping also keeps the reviews table from being dominated by 1-2 popular books.

**Why use two FK paths for reviews instead of always using books.id?**
Migrating the existing 5.5M reviews from isbn13-keyed to id-keyed would have been a large one-time operation with no upside for already-linked rows. The dual-path design adds zero overhead for the common case (isbn books) while unlocking reviews for the 288K no-isbn books. The RAG retrieval layer does not care which path was used - it just needs review text attached to a book record.
