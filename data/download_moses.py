#!/usr/bin/env python3
"""
data/download_moses.py

Downloads the official MOSES molecular benchmark dataset (dataset_v1.csv) to data/moses.csv.
Source: Molecular Sets (MOSES) benchmark (Polykovskiy et al., 2020)
https://github.com/molecularsets/moses
"""

import os
import sys
from pathlib import Path
import urllib.request

MOSES_URL = "https://media.githubusercontent.com/media/molecularsets/moses/master/data/dataset_v1.csv"
DATA_DIR = Path(__file__).resolve().parent
TARGET_FILE = DATA_DIR / "moses.csv"


def download_moses():
    if TARGET_FILE.exists() and TARGET_FILE.stat().st_size > 50 * 1024 * 1024:
        print(f"data/moses.csv already exists ({TARGET_FILE.stat().st_size / (1024*1024):.1f} MB). Skipping download.")
        return

    print(f"Downloading MOSES dataset from {MOSES_URL}...")
    tmp_file = TARGET_FILE.with_suffix(".csv.tmp")
    try:
        def reporthook(block_num, block_size, total_size):
            downloaded = block_num * block_size
            if total_size > 0:
                percent = min(100.0, downloaded * 100.0 / total_size)
                sys.stdout.write(f"\rProgress: {downloaded / (1024*1024):.1f} MB / {total_size / (1024*1024):.1f} MB ({percent:.1f}%)")
            else:
                sys.stdout.write(f"\rDownloaded: {downloaded / (1024*1024):.1f} MB")
            sys.stdout.flush()

        urllib.request.urlretrieve(MOSES_URL, tmp_file, reporthook=reporthook)
        print("\nDownload complete. Moving to data/moses.csv...")
        os.replace(tmp_file, TARGET_FILE)
        print(f"Saved {TARGET_FILE} ({TARGET_FILE.stat().st_size / (1024*1024):.1f} MB).")
    except Exception as e:
        if tmp_file.exists():
            tmp_file.unlink()
        print(f"\nError downloading MOSES dataset: {e}")
        sys.exit(1)


if __name__ == "__main__":
    download_moses()
