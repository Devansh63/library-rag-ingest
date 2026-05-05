"""
Embed books using BAAI/bge-base-en-v1.5 (local via sentence-transformers).

Two vectors per book:
  metadata_embedding  -- title + genres + synopsis
  review_embedding    -- KMeans medoids from reader reviews (k=5)

The medoid approach picks 5 representative real reviews instead of averaging
all of them, which would blur distinct reader perspectives into noise.

Usage:
    uv run python scripts/embedding/embed_sample.py
    uv run python scripts/embedding/embed_sample.py --dry-run --limit 100
    uv run python scripts/embedding/embed_sample.py --queued
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent.parent / ".env")

from lib.db import get_connection

BGE_MODEL        = "BAAI/bge-base-en-v1.5"
ENCODE_BATCH     = 64
KMEANS_K         = 5
MIN_REVIEW_CHARS = 50
MIN_REVIEWS      = 3
MEDOID_WORDS     = 80   # 5 medoids * 80 words stays under BGE's 512-token limit
SAMPLE_SIZE      = 1000
RANDOM_SEED      = 0.42  # setseed() takes float in [-1, 1]

QUEUED_MODE = False


def to_pg_vector(arr: np.ndarray) -> str:
    return "[" + ",".join(f"{x:.7f}" for x in arr.tolist()) + "]"


def select_sample_books(conn) -> list[dict]:
    print("Selecting sample books...")
    with conn.cursor() as cur:
        if QUEUED_MODE:
            # Process books pre-marked by mark_embed_queue.py, richest review
            # counts first so KMeans has the best signal early.
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
                    AND b.metadata_embedding IS NULL
                ORDER BY random()
                LIMIT %s
            """, (SAMPLE_SIZE,))

        cols = [d[0] for d in cur.description]
        books = [dict(zip(cols, row)) for row in cur.fetchall()]

    print(f"  Selected {len(books):,} books.")
    return books


def fetch_reviews(conn, books: list[dict]) -> dict[int, list[str]]:
    """Fetch all reviews for the given books in two queries (isbn path + book_id path)."""
    print("Fetching reviews...")
    book_ids  = [b["id"]    for b in books]
    isbn_list = [b["isbn13"] for b in books if b["isbn13"]]

    reviews: dict[int, list[str]] = {b["id"]: [] for b in books}
    isbn_to_id = {b["isbn13"]: b["id"] for b in books if b["isbn13"]}

    with conn.cursor() as cur:
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


def build_metadata_text(book: dict) -> str:
    """Build the string we embed for metadata_embedding.

    Format: 'Genres: {genres+subjects}. {title}. {description}'
    Description priority: plot_summary > synopsis > short_description.
    """
    parts = []
    genres   = book.get("genres")   or []
    subjects = book.get("subjects") or []
    if genres or subjects:
        parts.append("Genres: " + ", ".join(list(genres) + list(subjects)))

    description = (
        book.get("plot_summary") or
        book.get("synopsis") or
        book.get("short_description") or ""
    )

    return (". ".join(parts) + ". " + book["title"] + ". " + description).strip()


def filter_and_truncate_reviews(texts: list[str]) -> list[str]:
    """Drop short reviews and truncate to MEDOID_WORDS words."""
    return [
        " ".join(t.split()[:MEDOID_WORDS])
        for t in texts
        if len(t.strip()) >= MIN_REVIEW_CHARS
    ]


def select_medoids(embeddings: np.ndarray, texts: list[str], k: int) -> list[str]:
    """Return the real review closest to each KMeans centroid."""
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
        cluster_embs  = embeddings[mask]
        cluster_texts = [texts[i] for i, m in enumerate(mask) if m]
        centroid = km.cluster_centers_[cluster_id]
        dists = np.linalg.norm(cluster_embs - centroid, axis=1)
        medoids.append(cluster_texts[np.argmin(dists)])

    return medoids


def run(dry_run: bool = False) -> None:
    import torch
    from sentence_transformers import SentenceTransformer

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Device: {device}")
    if device == "cpu":
        print("  WARNING: MPS not available - this will be slow.")

    print(f"Loading {BGE_MODEL}...")
    t0 = time.time()
    model = SentenceTransformer(BGE_MODEL, device=device)
    print(f"  Loaded in {time.time() - t0:.1f}s")

    conn = get_connection()
    books = select_sample_books(conn)
    if not books:
        print("No qualifying books found. Run dedup_reviews.py first.")
        conn.close()
        return

    reviews_by_book = fetch_reviews(conn, books)

    # Phase 1: metadata embeddings
    print("\nPhase 1: Metadata embeddings...")
    meta_texts = [build_metadata_text(b) for b in books]
    t0 = time.time()
    meta_vecs = model.encode(
        meta_texts,
        batch_size=ENCODE_BATCH,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    print(f"  Done in {time.time() - t0:.1f}s -- {len(meta_vecs)} vectors")

    # Phase 2: review embeddings
    # Collect all individual reviews across all books into one flat list,
    # then do a single BGE pass. Slices track which range belongs to each book.
    print("\nPhase 2a: Collecting and filtering reviews...")
    all_review_texts: list[str] = []
    book_slices: dict[int, tuple[int, int]] = {}

    skipped_too_few = 0
    for book in books:
        bid = book["id"]
        raw = reviews_by_book.get(bid, [])
        filtered = filter_and_truncate_reviews(raw)
        if len(filtered) < MIN_REVIEWS:
            skipped_too_few += 1
            book_slices[bid] = (-1, -1)
            continue
        start = len(all_review_texts)
        all_review_texts.extend(filtered)
        book_slices[bid] = (start, len(all_review_texts))

    print(f"  {len(all_review_texts):,} reviews to encode.")
    print(f"  {skipped_too_few} books skipped (< {MIN_REVIEWS} usable reviews).")

    print("\nPhase 2b: Embedding individual reviews...")
    t0 = time.time()
    all_review_vecs = model.encode(
        all_review_texts,
        batch_size=ENCODE_BATCH,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    print(f"  Done in {time.time() - t0:.1f}s")

    print("\nPhase 2c: KMeans clustering and medoid selection...")
    medoid_texts_by_book: dict[int, str] = {}
    for book in books:
        bid = book["id"]
        start, end = book_slices[bid]
        if start == -1:
            continue
        medoids = select_medoids(all_review_vecs[start:end], all_review_texts[start:end], KMEANS_K)
        medoid_texts_by_book[bid] = " ".join(medoids)

    print(f"\nPhase 2d: Embedding {len(medoid_texts_by_book):,} medoid concatenations...")
    medoid_book_ids = list(medoid_texts_by_book.keys())
    t0 = time.time()
    review_vecs = model.encode(
        [medoid_texts_by_book[bid] for bid in medoid_book_ids],
        batch_size=ENCODE_BATCH,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    review_vec_by_book = dict(zip(medoid_book_ids, review_vecs))
    print(f"  Done in {time.time() - t0:.1f}s")

    if dry_run:
        print("\n[dry-run] Skipping DB writes.")
        _print_summary(books, meta_vecs, review_vec_by_book, skipped_too_few)
        conn.close()
        return

    # Reconnect - Neon drops idle connections after ~5 min and the encode
    # phases above can take longer than that.
    print("\nPhase 3: Writing embeddings to DB...")
    conn.close()
    conn = get_connection()
    from psycopg2.extras import execute_values

    rows = []
    for i, book in enumerate(books):
        bid = book["id"]
        review_vec = review_vec_by_book.get(bid)
        rows.append((
            bid,
            to_pg_vector(meta_vecs[i]),
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
    print(f"Books processed:          {len(books):,}")
    print(f"metadata_embedding set:   {len(meta_vecs):,}")
    print(f"review_embedding set:     {len(review_vec_by_book):,}")
    print(f"review_embedding skipped: {skipped_too_few} (< {MIN_REVIEWS} usable reviews)")
    print(f"Vector dimensions:        {meta_vecs[0].shape[0]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dry-run", action="store_true",
                        help="Run everything except the DB write.")
    parser.add_argument("--limit", type=int, default=SAMPLE_SIZE,
                        help="How many books to embed (default: 1000).")
    parser.add_argument("--queued", action="store_true",
                        help="Only process books marked EMBED_QUEUED.")
    args = parser.parse_args()
    SAMPLE_SIZE = args.limit
    QUEUED_MODE = args.queued
    run(dry_run=args.dry_run)
