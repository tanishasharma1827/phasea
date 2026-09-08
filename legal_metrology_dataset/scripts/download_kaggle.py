#!/usr/bin/env python3
"""
Download supplementary Kaggle datasets (barcode/receipt/product-label sets).

WHY THIS SCRIPT ISN'T ALREADY RUN FOR YOU:
kaggle.com and www.kaggle.com are unreachable from the sandbox this dataset
was built in (verified with a direct HTTPS probe -> HTTP 403). Run this on
your own machine.

Setup (one-time):
    1. Create a Kaggle account, go to Account -> Create New API Token.
    2. Save the downloaded kaggle.json to ~/.kaggle/kaggle.json (chmod 600).
    3. pip install kaggle

Usage:
    python download_kaggle.py
"""
import subprocess
import os

# Real, existing Kaggle dataset slugs found via web search on 2026-09-08.
# These are tangential (barcode/receipt OCR), not Legal-Metrology-labelled --
# see dataset_report.md for why none of these qualify as primary training data.
DATASETS = [
    "trainingdatapro/ocr-receipts-text-detection",
    "trainingdatapro/ocr-barcodes-detection",
    "dagloxkankwanda/package-design-dataset",
]

OUT_DIR = "./kaggle_download"


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    for slug in DATASETS:
        dest = os.path.join(OUT_DIR, slug.split("/")[-1])
        os.makedirs(dest, exist_ok=True)
        print(f"Downloading {slug} -> {dest}")
        try:
            subprocess.run(
                ["kaggle", "datasets", "download", "-d", slug, "-p", dest, "--unzip"],
                check=True,
            )
        except subprocess.CalledProcessError as e:
            print(f"  FAILED for {slug}: {e}. Check that the dataset still exists and "
                  f"your ~/.kaggle/kaggle.json is valid.")
    print("\nDone. Check each dataset's own license/terms on its Kaggle page before "
          "redistributing -- do not assume Kaggle-hosted = freely redistributable.")
    print("These are auxiliary/practice sets only; do not merge into the core "
          "legal_metrology_dataset/images tree without re-running the leakage-safe "
          "product-level split in scripts/build_dataset.py.")


if __name__ == "__main__":
    main()
