"""Comprehensive tests for the embedding sample.

Tests are grouped into four categories:

  1. INTEGRITY   - vectors exist, right dimensions, no NaN/Inf, normalized
  2. GEOMETRY    - cosine similarity distribution, intra-vs-inter genre separation
  3. SEMANTIC    - nearest-neighbor queries on known thematic phrases
  4. CONSISTENCY - metadata vs review embedding agreement per book

Run:
    uv run python scripts/embedding/test_embeddings.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent.parent / ".env")

from lib.db import get_connection

BGE_MODEL = "BAAI/bge-base-en-v1.5"
DIM = 768
PASS = "\033[32mPASS\033[0m"
FAIL = "\033[31mFAIL\033[0m"
WARN = "\033[33mWARN\033[0m"


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))


def check(label: str, condition: bool, detail: str = "") -> bool:
    status = PASS if condition else FAIL
    print(f"  [{status}] {label}" + (f" --- {detail}" if detail else ""))
    return condition


def test_integrity(meta: np.ndarray, review: np.ndarray, titles: list[str]) -> None:
    print("\n=== 1. INTEGRITY ===")

    check("All 1000 books have metadata_embedding", len(meta) == 1000, f"got {len(meta)}")

    review_present = np.sum(~np.all(review == 0, axis=1))
    check("Review embeddings present (expect ~1000)", review_present >= 950,
          f"{review_present}/1000 non-zero")

    check("Metadata dimension = 768", meta.shape[1] == DIM, f"got {meta.shape[1]}")
    check("Review dimension = 768",   review.shape[1] == DIM, f"got {review.shape[1]}")

    meta_nan   = np.any(np.isnan(meta)   | np.isinf(meta))
    review_nan = np.any(np.isnan(review) | np.isinf(review))
    check("No NaN/Inf in metadata embeddings", not meta_nan)
    check("No NaN/Inf in review embeddings",   not review_nan)

    meta_norms   = np.linalg.norm(meta,   axis=1)
    review_norms = np.linalg.norm(review, axis=1)
    check("Metadata vectors are unit-normalized", np.allclose(meta_norms,   1.0, atol=1e-4),
          f"mean norm={meta_norms.mean():.6f}")
    check("Review vectors are unit-normalized",   np.allclose(review_norms, 1.0, atol=1e-4),
          f"mean norm={review_norms.mean():.6f}")


def test_geometry(meta: np.ndarray, review: np.ndarray, genres: list[list]) -> None:
    print("\n=== 2. GEOMETRY ===")

    rng = np.random.default_rng(42)
    idx = rng.choice(len(meta), size=min(200, len(meta)), replace=False)
    sub = meta[idx]
    sims = sub @ sub.T
    np.fill_diagonal(sims, np.nan)
    flat = sims[~np.isnan(sims)]

    print(f"  Metadata cosine similarity (200-book sample):")
    print(f"    mean={flat.mean():.4f}  median={np.median(flat):.4f}  "
          f"min={flat.min():.4f}  max={flat.max():.4f}  std={flat.std():.4f}")
    check("Mean inter-book similarity < 0.95 (no collapse)", flat.mean() < 0.95, f"mean={flat.mean():.4f}")
    check("Mean inter-book similarity > 0.0 (not random noise)", flat.mean() > 0.0, f"mean={flat.mean():.4f}")

    genre_map: dict[str, list[int]] = {}
    for i, g_list in enumerate(genres):
        for g in (g_list or []):
            genre_map.setdefault(g, []).append(i)

    candidates = [(g, idxs) for g, idxs in genre_map.items() if len(idxs) >= 10]
    candidates.sort(key=lambda x: -len(x[1]))

    if len(candidates) >= 2:
        g1_name, g1_idx = candidates[0]
        for g2_name, g2_idx in candidates[1:]:
            if len(set(g1_idx) & set(g2_idx)) < 3:
                break

        g1_vecs = meta[g1_idx[:15]]
        g2_vecs = meta[g2_idx[:15]]
        within_g1 = g1_vecs @ g1_vecs.T
        np.fill_diagonal(within_g1, np.nan)
        within_mean = np.nanmean(within_g1)
        across_mean = (g1_vecs @ g2_vecs.T).mean()

        print(f"\n  Genre separation: '{g1_name}' vs '{g2_name}'")
        print(f"    within-genre: {within_mean:.4f}  across-genre: {across_mean:.4f}")
        check("Within-genre similarity > across-genre", within_mean > across_mean,
              f"{within_mean:.4f} > {across_mean:.4f}")
    else:
        print(f"  {WARN} Not enough genre data for separation test")


def test_semantic(meta, review, titles, genres) -> None:
    print("\n=== 3. SEMANTIC (nearest-neighbor queries) ===")

    import torch
    from sentence_transformers import SentenceTransformer
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = SentenceTransformer(BGE_MODEL, device=device)

    queries = [
        ("dark atmospheric fantasy",         ["fantasy", "fiction", "dark"]),
        ("detective mystery crime thriller",  ["mystery", "thriller", "crime", "detective"]),
        ("coming of age young adult",         ["young-adult", "fiction", "ya", "teen"]),
        ("historical fiction war",            ["historical", "war", "history", "fiction"]),
        ("romance love story",                ["romance", "love", "fiction"]),
        ("science fiction space exploration", ["science-fiction", "sci-fi", "space", "fiction"]),
    ]

    for query_text, expected_genres in queries:
        qvec = model.encode(query_text, normalize_embeddings=True)
        meta_scores   = meta   @ qvec
        review_scores = review @ qvec
        top5_meta   = np.argsort(meta_scores)[::-1][:5]
        top5_review = np.argsort(review_scores)[::-1][:5]

        def genre_hit(top_idxs):
            for i in top_idxs:
                for g in (genres[i] or []):
                    if any(eg in g.lower() for eg in expected_genres):
                        return True
            return False

        print(f"\n  Query: '{query_text}'")
        for i in top5_meta:
            g = ", ".join((genres[i] or [])[:3])
            print(f"    meta   [{meta_scores[i]:.4f}] {titles[i][:55]}  [{g}]")
        for i in top5_review:
            g = ", ".join((genres[i] or [])[:3])
            print(f"    review [{review_scores[i]:.4f}] {titles[i][:55]}  [{g}]")

        check(f"Metadata genre hit for '{query_text}'", genre_hit(top5_meta))
        check(f"Review   genre hit for '{query_text}'", genre_hit(top5_review))


def test_consistency(meta, review, titles) -> None:
    print("\n=== 4. CONSISTENCY (metadata vs review agreement) ===")

    per_book_sim = np.sum(meta * review, axis=1)
    print(f"  Per-book metadata<->review cosine:")
    print(f"    mean={per_book_sim.mean():.4f}  median={np.median(per_book_sim):.4f}  "
          f"min={per_book_sim.min():.4f}  max={per_book_sim.max():.4f}")
    check("Mean self-similarity > 0.2", per_book_sim.mean() > 0.2, f"mean={per_book_sim.mean():.4f}")

    meta_top10   = np.argsort(-(meta   @ meta.T),   axis=1)[:, 1:11]
    review_top10 = np.argsort(-(review @ review.T), axis=1)[:, 1:11]
    mean_overlap = np.mean([len(set(meta_top10[i]) & set(review_top10[i])) for i in range(len(titles))])
    print(f"\n  Top-10 neighbor overlap: mean={mean_overlap:.2f}/10")
    check("Mean neighbor overlap > 1.0", mean_overlap > 1.0, f"mean={mean_overlap:.2f}")

    bottom5 = np.argsort(per_book_sim)[:5]
    print("\n  Lowest metadata<->review agreement books:")
    for i in bottom5:
        print(f"    [{per_book_sim[i]:.4f}] {titles[i][:70]}")


def main() -> None:
    t_start = time.time()
    print("Connecting to DB...")
    conn = get_connection()

    print("Fetching embeddings for 1000 sampled books...")
    with conn.cursor() as cur:
        cur.execute("""
            SELECT b.title, b.genres,
                   b.metadata_embedding::text,
                   b.review_embedding::text
            FROM books b
            WHERE b.metadata_embedding IS NOT NULL
            ORDER BY b.id
            LIMIT 1000
        """)
        rows = cur.fetchall()
    conn.close()

    print(f"  Fetched {len(rows)} rows.")

    def parse_vec(s):
        if s is None:
            return np.zeros(DIM)
        return np.array([float(x) for x in s.strip("[]").split(",")], dtype=np.float32)

    titles = [r[0] for r in rows]
    genres = [list(r[1]) if r[1] else [] for r in rows]
    meta   = np.array([parse_vec(r[2]) for r in rows])
    review = np.array([parse_vec(r[3]) for r in rows])

    test_integrity(meta, review, titles)
    test_geometry(meta, review, genres)
    test_semantic(meta, review, titles, genres)
    test_consistency(meta, review, titles)

    print(f"\nTotal test time: {time.time() - t_start:.1f}s")


if __name__ == "__main__":
    main()
