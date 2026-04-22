"""
Download the three primary datasets used by the CS 410 Group 3 project.

Datasets and sources:
    1. GoodReads Best Books Ever (Zenodo) - ~52k books, metadata-rich.
       Direct download from Zenodo; no login required.
    2. CMU Book Summary Dataset - ~16.5k books with plot summaries.
       Direct download from CMU; no login required.
    3. UCSD Goodreads Book Graph - reviews + nested genres.
       Hosted on UCSD's site. Several files; we grab the two we need for
       ingestion (books metadata JSON and reviews JSON). These files are
       large (multi-GB compressed) so we download them only if requested
       via --ucsd flag, and we stream them rather than buffering in memory.

Usage:
    uv run python scripts/download_datasets.py           # Zenodo + CMU only
    uv run python scripts/download_datasets.py --ucsd    # All three
    uv run python scripts/download_datasets.py --dry-run # Print URLs, no download
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path
from typing import NamedTuple

import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

import httpx
import gdown
from tqdm import tqdm


# All paths relative to the project root data/ directory.
DATA_DIR = Path(__file__).parent.parent / "data" / "raw"


class DatasetFile(NamedTuple):
    name: str            # Human-readable label for progress display.
    url: str             # Download URL (direct HTTP or Google Drive uc?id= URL).
    dest: str            # Filename under data/raw/.
    sha256: str | None   # Optional expected hash.
    large: bool          # If True, print a size warning before downloading.
    gdrive: bool = False # If True, use gdown instead of httpx (Google Drive files).


# --- Dataset definitions ---

ZENODO_FILES = [
    DatasetFile(
        name="GoodReads Best Books Ever (Zenodo)",
        url="https://zenodo.org/records/4265096/files/books_1.Best_Books_Ever.csv",
        dest="goodreads_bbe.csv",
        sha256=None,
        large=False,
    ),
]

CMU_FILES = [
    DatasetFile(
        name="CMU Book Summaries - plot summaries",
        url="http://www.cs.cmu.edu/~dbamman/data/booksummaries.tar.gz",
        dest="cmu_booksummaries.tar.gz",
        sha256=None,
        large=False,
    ),
]

# UCSD dataset direct URLs from the paper's cited page:
# https://cseweb.ucsd.edu/~jmcauley/datasets/goodreads.html
# These are direct HTTPS downloads from mcauleylab.ucsd.edu - no Google Drive needed.
UCSD_FILES = [
    DatasetFile(
        name="UCSD Goodreads - books (metadata + genres)",
        url="https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/goodreads_books.json.gz",
        dest="ucsd_goodreads_books.json.gz",
        sha256=None,
        large=True,
        gdrive=False,
    ),
    DatasetFile(
        name="UCSD Goodreads - reviews (deduplicated)",
        url="https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/goodreads_reviews_dedup.json.gz",
        dest="ucsd_goodreads_reviews.json.gz",
        sha256=None,
        large=True,
        gdrive=False,
    ),
]


def sha256_of_file(path: Path) -> str:
    """Compute the SHA-256 digest of a file in chunks, without loading it all."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download_file(dataset: DatasetFile, dest_dir: Path, dry_run: bool) -> bool:
    """Download one file. Returns True on success, False on failure.

    Skips the download if the destination file already exists and (when a
    hash is available) matches the expected SHA-256. This makes re-runs safe
    and cheap.
    """
    dest_path = dest_dir / dataset.dest

    # Skip if already present and valid.
    if dest_path.exists():
        if dataset.sha256:
            actual = sha256_of_file(dest_path)
            if actual == dataset.sha256:
                print(f"  [skip] {dataset.dest} already exists and hash matches.")
                return True
            else:
                print(
                    f"  [redownload] {dataset.dest} exists but hash mismatch "
                    f"(expected {dataset.sha256[:8]}... got {actual[:8]}...)."
                )
        else:
            print(f"  [skip] {dataset.dest} already exists (no hash to verify).")
            return True

    if dataset.large:
        print(
            f"  WARNING: {dataset.name} is a large file (potentially several GB "
            f"compressed). Make sure you have enough disk space."
        )

    if dry_run:
        print(f"  [dry-run] Would download: {dataset.url}")
        print(f"            -> {dest_path}")
        return True

    print(f"  Downloading: {dataset.name}")
    print(f"    from: {dataset.url}")
    print(f"    to:   {dest_path}")

    if dataset.gdrive:
        # Google Drive files need gdown to handle the virus-scan confirmation
        # redirect that Google adds for large files (httpx would get an HTML page).
        try:
            result = gdown.download(dataset.url, str(dest_path), quiet=False, fuzzy=True)
            if not result or not dest_path.exists():
                print(f"  ERROR: gdown returned no output for {dataset.url}", file=sys.stderr)
                dest_path.unlink(missing_ok=True)
                return False
        except Exception as exc:
            print(f"  ERROR: {exc}", file=sys.stderr)
            dest_path.unlink(missing_ok=True)
            return False
    else:
        try:
            with httpx.stream("GET", dataset.url, follow_redirects=True, timeout=300) as resp:
                resp.raise_for_status()

                total = int(resp.headers.get("content-length", 0)) or None
                with (
                    dest_path.open("wb") as out_f,
                    tqdm(
                        total=total,
                        unit="B",
                        unit_scale=True,
                        unit_divisor=1024,
                        desc=dataset.dest,
                        leave=True,
                    ) as bar,
                ):
                    for chunk in resp.iter_bytes(chunk_size=1 << 16):
                        out_f.write(chunk)
                        bar.update(len(chunk))

        except httpx.HTTPStatusError as exc:
            print(f"  ERROR: HTTP {exc.response.status_code} for {dataset.url}", file=sys.stderr)
            dest_path.unlink(missing_ok=True)
            return False
        except Exception as exc:
            print(f"  ERROR: {exc}", file=sys.stderr)
            dest_path.unlink(missing_ok=True)
            return False

    if dataset.sha256:
        actual = sha256_of_file(dest_path)
        if actual != dataset.sha256:
            print(
                f"  ERROR: Hash mismatch after download. Expected {dataset.sha256}, "
                f"got {actual}.",
                file=sys.stderr,
            )
            return False
        print(f"  Hash verified.")

    size_mb = dest_path.stat().st_size / (1 << 20)
    print(f"  Done. ({size_mb:.1f} MB)")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--ucsd",
        action="store_true",
        help="Also download the large UCSD Goodreads Book Graph files.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be downloaded without actually downloading anything.",
    )
    args = parser.parse_args()

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    files_to_download = list(ZENODO_FILES) + list(CMU_FILES)
    if args.ucsd:
        files_to_download.extend(UCSD_FILES)
    else:
        print(
            "Note: UCSD Goodreads Book Graph files are skipped by default "
            "(multi-GB). Pass --ucsd to include them.\n"
        )

    failures: list[str] = []
    for dataset in files_to_download:
        print(f"\n[{dataset.name}]")
        success = download_file(dataset, DATA_DIR, dry_run=args.dry_run)
        if not success:
            failures.append(dataset.name)

    print()
    if failures:
        print(f"Failed downloads ({len(failures)}):")
        for name in failures:
            print(f"  - {name}")
        print(
            "\nFor failed files, try downloading manually and placing the file "
            f"in {DATA_DIR}/ with the expected filename."
        )
        return 1

    print("All downloads complete.")
    print(f"Files are in: {DATA_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
