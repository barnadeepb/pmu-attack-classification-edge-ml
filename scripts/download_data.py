"""
Fetches the Mississippi State University / Oak Ridge National Laboratory
"Power System Attack Datasets" (3-class version: Natural / Attack / NoEvents)
and combines the 15 source files into a single CSV under data/.

Source: Tommy Morris, University of Alabama in Huntsville
  https://sites.google.com/a/uah.edu/tommy-morris-uah/ics-data-sets
  http://www.ece.uah.edu/~thm0009/icsdatasets/triple.7z

If the download fails (the host is an old university server and does go
down occasionally), grab triple.7z manually from the URL above, drop it in
data/raw/, and re-run this script -- it will detect the local archive and
skip straight to extraction.
"""

import sys
from pathlib import Path

import pandas as pd
import py7zr
import requests

DATA_URL = "http://www.ece.uah.edu/~thm0009/icsdatasets/triple.7z"
# The original UAH host returned HTTP 404 when re-checked in October 2026.
# The Internet Archive holds a snapshot of the same URL; it is tried second
# and size-checked against the archive used for the original experiments.
MIRROR_URL = "https://web.archive.org/web/2022if_/http://www.ece.uah.edu/~thm0009/icsdatasets/triple.7z"
EXPECTED_ARCHIVE_BYTES = 19_701_721

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
ARCHIVE_PATH = RAW_DIR / "triple.7z"
EXTRACT_DIR = RAW_DIR / "extracted"
OUTPUT_PATH = DATA_DIR / "pmu_dataset.csv"

EXPECTED_FILES = [f"data{i}.csv" for i in range(1, 16)]
EXPECTED_COLUMNS = 129  # 128 features + marker


def _try_download(url):
    try:
        with requests.get(url, stream=True, timeout=60) as response:
            response.raise_for_status()
            total = int(response.headers.get("content-length", 0))
            written = 0
            with open(ARCHIVE_PATH, "wb") as f:
                for chunk in response.iter_content(chunk_size=1 << 16):
                    f.write(chunk)
                    written += len(chunk)
                    if total:
                        pct = 100 * written / total
                        print(f"\r  {written / 1e6:6.1f} MB / {total / 1e6:6.1f} MB ({pct:5.1f}%)", end="")
        print("\nDownload complete.")
    except requests.RequestException as exc:
        if ARCHIVE_PATH.exists():
            ARCHIVE_PATH.unlink()
        print(f"\nDownload from {url} failed: {exc}")
        return False

    if ARCHIVE_PATH.stat().st_size != EXPECTED_ARCHIVE_BYTES:
        print(f"Unexpected archive size {ARCHIVE_PATH.stat().st_size} bytes; discarding.")
        ARCHIVE_PATH.unlink()
        return False
    return True


def download_archive():
    if ARCHIVE_PATH.exists():
        print(f"Found existing archive at {ARCHIVE_PATH}, skipping download.")
        return

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    for url in (DATA_URL, MIRROR_URL):
        print(f"Downloading dataset from {url} ...")
        if _try_download(url):
            return

    print(f"All downloads failed. Manually download {DATA_URL} and save it to:")
    print(f"  {ARCHIVE_PATH}")
    print("Then re-run this script.")
    sys.exit(1)


def extract_archive():
    if EXTRACT_DIR.exists() and all((EXTRACT_DIR / f).exists() for f in EXPECTED_FILES):
        print(f"Found existing extracted files in {EXTRACT_DIR}, skipping extraction.")
        return

    print(f"Extracting {ARCHIVE_PATH} ...")
    EXTRACT_DIR.mkdir(parents=True, exist_ok=True)
    with py7zr.SevenZipFile(ARCHIVE_PATH, mode="r") as archive:
        archive.extractall(path=EXTRACT_DIR)

    missing = [f for f in EXPECTED_FILES if not (EXTRACT_DIR / f).exists()]
    if missing:
        print(f"Extraction did not produce the expected files: {missing}")
        sys.exit(1)
    print("Extraction complete.")


def combine_files():
    print("Combining 15 source files into a single dataset ...")
    frames = []
    for file_no, name in enumerate(EXPECTED_FILES, start=1):
        df = pd.read_csv(EXTRACT_DIR / name)
        if df.shape[1] != EXPECTED_COLUMNS:
            print(f"WARNING: {name} has {df.shape[1]} columns, expected {EXPECTED_COLUMNS}.")
        # Keep the scenario-file index so experiments can split by scenario
        # (leave-one-scenario-out) instead of only by random rows. It is a
        # bookkeeping column and must never be used as a model feature.
        df["source_file"] = file_no
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)

    if "marker" not in combined.columns:
        print("ERROR: combined dataset has no 'marker' label column.")
        sys.exit(1)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    combined.to_csv(OUTPUT_PATH, index=False)

    print(f"\nSaved combined dataset to {OUTPUT_PATH}")
    print(f"Shape: {combined.shape[0]} rows x {combined.shape[1]} columns (incl. source_file)")
    print("Class distribution:")
    counts = combined["marker"].value_counts()
    for label, count in counts.items():
        print(f"  {label:<10} {count:>6}  ({100 * count / len(combined):5.1f}%)")
    print("Per source file class counts:")
    print(pd.crosstab(combined["source_file"], combined["marker"]).to_string())


if __name__ == "__main__":
    download_archive()
    extract_archive()
    combine_files()
    print("\nDone. Next: python scripts/run_experiment.py")
