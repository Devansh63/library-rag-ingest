"""
Test embedding pipeline on 1000 randomly sampled quality books.

WHY THIS EXISTS:
Before committing to a full 300K overnight run, we validate the entire
pipeline end-to-end: book selection, metadata text construction, review
clustering, and DB writes. 1000 books is fast enough to iterate on but
large enough to catch real edge cases.

TWO EMBEDDINGS PER BOOK:
  metadata_embedding  - BGE vector built from title + genres + synopsis.
                        Captures what the book IS (publisher language).
  review_embedding    - BGE vector built from KMeans-clustered reader review
                        snippets. Captures what readers SAY about the book
                        (reader language). Bridges the vocabulary gap between
                        how books are described vs how users search.

BOOK SELECTION (1000 random from ~78K quality books):
  - source = goodreads_bbe OR cmu_summaries  (always high quality)
  - source = ucsd_graph AND CLEANED flag     (ISBNdb-enriched UCSD books)
  - has at least one description field        (synopsis / short_desc / plot)
  - has 30+ reviews in DB                    (enough for meaningful KMeans)

HOW REVIEW EMBEDDING WORKS:
  1. Fetch up to 60 reviews per book (already capped during ingest)
  2. Filter: keep reviews >= 50 chars (removes noise ratings like "great!")
  3. Embed each review individually through BGE
  4. KMeans into k=5 clusters (finds natural "perspective" groupings)
  5. Pick the medoid (most representative real review) from each cluster
  6. Concatenate the 5 medoid snippets (truncated to 80 words each)
  7. Embed the concatenation once -> review_embedding

  The medoid approach preserves distinct reader perspectives rather than
  averaging them into a mushy centroid. A book about war might cluster into
  "devastating", "inspiring", "historical accuracy", "writing style", and
  "pacing" - averaging those would lose all signal.

Run AFTER dedup_reviews.py completes so review counts are accurate.

FIRST RUN: downloads BAAI/bge-base-en-v1.5 (~440 MB) from HuggingFace.

Usage:
    uv run python scripts/embedding/embed_sample.py
    uv run python scripts/embedding/embed_sample.py --dry-run
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans

# Project root on path so lib.* imports work from scripts/embedding/
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent.parent / ".env")

from lib.db import get_connection

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BGE_MODEL        = "BAAI/bge-base-en-v1.5"
ENCODE_BATCH     = 64    # BGE batch size - tuned for MPS throughput
KMEANS_K         = 5     # review clusters per book
MIN_REVIEW_CHARS = 50    # shorter reviews are noise, skip them
MIN_REVIEWS      = 3     # minimum usable reviews after char filter
MEDOID_WORDS     = 80    # truncate each medoid snippet before concat
                         # 5 medoids * 80 words = ~400 tokens, safely under BGE's 512 limit
SAMPLE_SIZE      = 1000  # overridden by --limit
RANDOM_SEED      = 0.42  # PostgreSQL setseed() takes a float in [-1, 1]


# ---------------------------------------------------------------------------
# Vector serialization
# ---------------------------------------------------------------------------

def to_pg_vector(arr: np.ndarray) -> str:
    """Format a numpy array as a pgvector literal: '[x,y,z,...]'.

    pgvector accepts this string format and casts it with ::vector(768).
    We use 7 decimal places - more than enough precision for cosine search.
    """
    return "[" + ",".join(f"{x:.7f}" for x in arr.tolist()) + "]"


# ---------------------------------------------------------------------------
# Book selection
# ---------------------------------------------------------------------------

QUEUED_MODE = False  # set to True via --queued flag


def select_sample_books(conn) -> list[dict]:
    """Select books to embed.

    In QUEUED_MODE: picks books marked 'EMBED_QUEUED' by mark_embed_queue.py,
    ordered by review count descending (richest KMeans signal first). This is
    the mode used for deliberate multi-machine runs where books are pre-scored
    and marked on the MacBook, then any machine processes the queue.

    In default mode: selects SAMPLE_SIZE random quality books with 30+ reviews.
    Uses setseed() for reproducible random() ordering across re-runs.

    Review counts are pre-aggregated in CTEs to avoid a correlated subquery
    scanning 11M rows once per candidate book.
    """
    print("Selecting sample books...")
    with conn.cursor() as cur:
        if QUEUED_MODE:
            cur.execute("""
                WITH
                isbn_counts AS (
                    SELECT isbn13, COUNT(*) AS cnt
                    FROM reviews WHERE isbn13 IS NOT NULL GROUP BY isbn13
                ),
                id_counts AS (
                    SELECT book_id, COUNT(*) AS cnt
                    FROM reviews WHERE book_id IS NOT NULL GROUP BY book_id
                )
                SELECT
                    b.id, b.title, b.isbn13, b.genres, b.subjects,
                    b.synopsis, b.short_description, b.plot_summary
                FROM books b
                LEFT JOIN isbn_counts ic  ON ic.isbn13   = b.isbn13
                LEFT JOIN id_counts   idc ON idc.book_id = b.id
                WHERE
                    'EMBED_QUEUED' = ANY(b.cleaning_flags)
                    AND b.metadata_embedding IS NULL
                ORDER BY (COALESCE(ic.cnt, 0) + COALESCE(idc.cnt, 0)) DESC
                LIMIT %s
            """, (SAMPLE_SIZE,))
        else:
            cur.execute("SELECT setseed(%s)", (RANDOM_SEED,))
            cur.execute("""
                WITH
                isbn_counts AS (
                    SELECT isbn13, COUNT(*) AS cnt
                    FROM reviews
                    WHERE isbn13 IS NOT NULL
                    GROUP BY isbn13
                ),
                id_counts AS (
                    SELECT book_id, COUNT(*) AS cnt
                    FROM reviews
                    WHERE book_id IS NOT NULL
                    GROUP BY book_id
                )
                SELECT
                    b.id,
                    b.title,
                    b.isbn13,
                    b.genres,
                    b.subjects,
                    b.synopsis,
                    b.short_description,
                    b.plot_summary
                FROM books b
                LEFT JOIN isbn_counts ic ON ic.isbn13 = b.isbn13
                LEFT JOIN id_counts   idc ON idc.book_id = b.id
                WHERE
                    (
                        b.source IN ('goodreads_bbe', 'cmu_summaries')
                        OR (b.source = 'ucsd_graph' AND 'CLEANED' = ANY(b.cleaning_flags))
                    )
                    AND (
                        b.synopsis          IS NOT NULL OR
                        b.short_description IS NOT NULL OR
                        b.plot_summary      IS NOT NULL
                    )
                    AND (COALESCE(ic.cnt, 0) + COALESCE(idc.cnt, 0)) >= 30
                    AND b.metadata_embedding IS NULL  -- skip already-embedded books
                ORDER BY random()
                LIMIT %s
            """, (SAMPLE_SIZE,))

        cols = [d[0] for d in cur.description]
        books = [dict(zip(cols, row)) for row in cur.fetchall()]

    print(f"  Selected {len(books):,} books.")
    return books


# ---------------------------------------------------------------------------
# Review fetching
# ---------------------------------------------------------------------------

def fetch_reviews(conn, books: list[dict]) -> dict[int, list[str]]:
    """Fetch reviews for all sample books in one query.

    Returns {book_id -> [review_text, ...]}. Already filtered to non-empty
    text; char-length filtering happens later during embedding prep so we
    can report how many get dropped.
    """
    print("Fetching reviews...")
    book_ids   = [b["id"]    for b in books]
    isbn_list  = [b["isbn13"] for b in books if b["isbn13"]]

    reviews: dict[int, list[str]] = {b["id"]: [] for b in books}

    # Build isbn13 -> book_id lookup for routing isbn-path reviews back
    isbn_to_id = {b["isbn13"]: b["id"] for b in books if b["isbn13"]}

    with conn.cursor() as cur:
        # isbn13 path
        if isbn_list:
            cur.execute("""
                SELECT isbn13, review_text
                FROM reviews
                WHERE isbn13 = ANY(%s)
                  AND review_text IS NOT NULL AND review_text <> ''
            """, (isbn_list,))
            for isbn13, text in cur.fetchall():
                book_id = isbn_to_id.get(isbn13)
                if book_id:
                    reviews[book_id].append(text)

        # book_id path (no-isbn books)
        cur.execute("""
            SELECT book_id, review_text
            FROM reviews
            WHERE book_id = ANY(%s)
              AND review_text IS NOT NULL AND review_text <> ''
        """, (book_ids,))
        for book_id, text in cur.fetchall():
            if book_id in reviews:
                reviews[book_id].append(text)

    total = sum(len(v) for v in reviews.values())
    print(f"  Fetched {total:,} reviews across {len(reviews):,} books.")
    return reviews


# ---------------------------------------------------------------------------
# Text construction
# ---------------------------------------------------------------------------

def build_metadata_text(book: dict) -> str:
    """Build the metadata embedding input string for one book.

    Template: 'Genres: {genres}, {subjects}. {title}. {description}'

    Description priority: plot_summary > synopsis > short_description.
    BGE truncates to 512 tokens automatically - no manual truncation needed
    here. CMU plot_summaries can be long but BGE handles it.
    """
    parts = []
    genres   = book.get("genres")   or []
    subjects = book.get("subjects") or []
    if genres or subjects:
        combined = list(genres) + list(subjects)
        parts.append("Genres: " + ", ".join(combined))

    description = (
        book.get("plot_summary")      or
        book.get("synopsis")          or
        book.get("short_description") or ""
    )

    text = ". ".join(parts) + ". " + book["title"] + ". " + description
    return text.strip()


def filter_and_truncate_reviews(texts: list[str]) -> list[str]:
    """Keep reviews >= MIN_REVIEW_CHARS and truncate to MEDOID_WORDS words.

    Truncation here is for the medoid concatenation step. Individual reviews
    are embedded at full length (BGE handles truncation), but when we
    concatenate 5 medoids we need to stay under 512 tokens total.
    """
    return [
        " ".join(t.split()[:MEDOID_WORDS])
        for t in texts
        if len(t.strip()) >= MIN_REVIEW_CHARS
    ]


# ---------------------------------------------------------------------------
# KMeans medoid selection
# ---------------------------------------------------------------------------

def select_medoids(
    embeddings: np.ndarray,
    texts: list[str],
    k: int,
) -> list[str]:
    """Run KMeans and return the medoid text from each cluster.

    The medoid is the real review whose embedding is closest to its cluster
    centroid. We return the actual text (not the vector) so we can embed
    the concatenation as a single unit.

    Why medoid not centroid: the centroid is a mathematical average in
    vector space with no corresponding real text. We need real text to feed
    back into BGE for the final review_embedding.
    """
    k = min(k, len(embeddings))
    if k == 1:
        return [texts[0]]

    km = KMeans(n_clusters=k, random_state=42, n_init="auto")
    km.fit(embeddings)

    medoids = []
    for cluster_id in range(k):
        mask = km.labels_ == cluster_id
        if not mask.any():
            continue
        cluster_embs = embeddings[mask]
        cluster_texts = [texts[i] for i, m in enumerate(mask) if m]
        centroid = km.cluster_centers_[cluster_id]
        dists = np.linalg.norm(cluster_embs - centroid, axis=1)
        medoids.append(cluster_texts[np.argmin(dists)])

    return medoids


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run(dry_run: bool = False) -> None:
    import torch
    from sentence_transformers import SentenceTransformer

    # ---- Device setup -------------------------------------------------------
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Device: {device}")
    if device == "cpu":
        print("  WARNING: MPS not available - embedding will be slow.")

    # ---- Load model ---------------------------------------------------------
    print(f"Loading {BGE_MODEL}...")
    t0 = time.time()
    model = SentenceTransformer(BGE_MODEL, device=device)
    print(f"  Loaded in {time.time() - t0:.1f}s")

    # ---- DB + data ----------------------------------------------------------
    conn = get_connection()
    books = select_sample_books(conn)
    if not books:
        print("No qualifying books found. Check that dedup_reviews.py has run.")
        conn.close()
        return

    reviews_by_book = fetch_reviews(conn, books)

    # ---- Phase 1: Metadata embeddings (one batch pass) ---------------------
    print("\nPhase 1: Metadata embeddings...")
    meta_texts = [build_metadata_text(b) for b in books]
    t0 = time.time()
    meta_vecs = model.encode(
        meta_texts,
        batch_size=ENCODE_BATCH,
        show_progress_bar=True,
        normalize_embeddings=True,  # cosine similarity works on normalized vecs
    )
    print(f"  Done in {time.time() - t0:.1f}s -- {len(meta_vecs)} vectors")

    # ---- Phase 2: Review embeddings ----------------------------------------
    # Step 2a: collect ALL individual review texts across all books in one list,
    # tracking where each book's reviews start and end. This lets us do one big
    # BGE batch pass instead of one pass per book - much better GPU utilization.
    print("\nPhase 2a: Collecting and filtering reviews...")
    all_review_texts: list[str] = []
    book_slices: dict[int, tuple[int, int]] = {}  # book_id -> (start, end) in all_review_texts

    skipped_too_few = 0
    for book in books:
        bid = book["id"]
        raw = reviews_by_book.get(bid, [])
        filtered = filter_and_truncate_reviews(raw)
        if len(filtered) < MIN_REVIEWS:
            skipped_too_few += 1
            book_slices[bid] = (-1, -1)  # sentinel: skip this book's review embedding
            continue
        start = len(all_review_texts)
        all_review_texts.extend(filtered)
        book_slices[bid] = (start, len(all_review_texts))

    print(f"  {len(all_review_texts):,} individual reviews to encode.")
    print(f"  {skipped_too_few} books skipped (< {MIN_REVIEWS} usable reviews after filtering).")

    # Step 2b: embed all individual reviews in one pass
    print("\nPhase 2b: Embedding individual reviews...")
    t0 = time.time()
    all_review_vecs = model.encode(
        all_review_texts,
        batch_size=ENCODE_BATCH,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    print(f"  Done in {time.time() - t0:.1f}s")

    # Step 2c: KMeans per book, collect medoid texts
    print("\nPhase 2c: KMeans clustering and medoid selection...")
    medoid_texts_by_book: dict[int, str] = {}
    for book in books:
        bid = book["id"]
        start, end = book_slices[bid]
        if start == -1:
            continue
        vecs  = all_review_vecs[start:end]
        texts = all_review_texts[start:end]
        medoids = select_medoids(vecs, texts, KMEANS_K)
        medoid_texts_by_book[bid] = " ".join(medoids)

    # Step 2d: embed the concatenated medoid strings - one per book
    print(f"\nPhase 2d: Embedding {len(medoid_texts_by_book):,} medoid concatenations...")
    medoid_book_ids = list(medoid_texts_by_book.keys())
    medoid_inputs   = [medoid_texts_by_book[bid] for bid in medoid_book_ids]
    t0 = time.time()
    review_vecs = model.encode(
        medoid_inputs,
        batch_size=ENCODE_BATCH,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    review_vec_by_book = dict(zip(medoid_book_ids, review_vecs))
    print(f"  Done in {time.time() - t0:.1f}s")

    # ---- Phase 3: Write to DB ----------------------------------------------
    if dry_run:
        print("\n[dry-run] Skipping DB writes.")
        _print_summary(books, meta_vecs, review_vec_by_book, skipped_too_few)
        conn.close()
        return

    # Reconnect before writing - the original connection may have timed out
    # during the long embedding phases (Neon drops idle connections after ~5 min).
    print("\nPhase 3: Writing embeddings to DB...")
    conn.close()
    conn = get_connection()
    from psycopg2.extras import execute_values

    rows = []
    for i, book in enumerate(books):
        bid = book["id"]
        meta_vec    = meta_vecs[i]
        review_vec  = review_vec_by_book.get(bid)  # None if skipped
        rows.append((
            bid,
            to_pg_vector(meta_vec),
            to_pg_vector(review_vec) if review_vec is not None else None,
        ))

    t0 = time.time()
    with conn.cursor() as cur:
        execute_values(cur, """
            UPDATE books AS b SET
                metadata_embedding = v.meta_emb::vector(768),
                review_embedding   = CASE
                    WHEN v.review_emb IS NOT NULL
                    THEN v.review_emb::vector(768)
                    ELSE b.review_embedding
                END
            FROM (VALUES %s) AS v(id, meta_emb, review_emb)
            WHERE b.id = v.id::integer
        """, rows)
    conn.commit()
    print(f"  Updated {len(rows):,} rows in {time.time() - t0:.1f}s.")

    _print_summary(books, meta_vecs, review_vec_by_book, skipped_too_few)
    conn.close()


def _print_summary(books, meta_vecs, review_vec_by_book, skipped_too_few):
    print("\n--- Summary ---")
    print(f"Books processed:         {len(books):,}")
    print(f"metadata_embedding set:  {len(meta_vecs):,}")
    print(f"review_embedding set:    {len(review_vec_by_book):,}")
    print(f"review_embedding skipped:{skipped_too_few} (< {MIN_REVIEWS} usable reviews)")
    print(f"Vector dimensions:       {meta_vecs[0].shape[0]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dry-run", action="store_true",
                        help="Run everything except the DB write.")
    parser.add_argument("--limit", type=int, default=SAMPLE_SIZE,
                        help="How many books to embed in this run (default: 1000).")
    parser.add_argument("--queued", action="store_true",
                        help="Only process books marked EMBED_QUEUED by mark_embed_queue.py.")
    args = parser.parse_args()
    SAMPLE_SIZE = args.limit
    QUEUED_MODE = args.queued
    run(dry_run=args.dry_run)
