#!/usr/bin/env python3
"""
Download public Roboflow Universe datasets (e.g. expiry-date detection sets).

WHY THIS SCRIPT ISN'T ALREADY RUN FOR YOU:
universe.roboflow.com and api.roboflow.com are unreachable from the sandbox
this dataset was built in (verified with a direct HTTPS probe -> HTTP 403).
Run this on your own machine, where you have normal internet access.

You will be prompted to log in / provide a free Roboflow API key on first run
(roboflow.login()). Each dataset has its own license (mostly CC BY 4.0, but
verify per-dataset on its Universe page before using it beyond a college project).

Usage:
    pip install roboflow
    python download_roboflow.py

Edit the DATASETS list below with the exact dataset_url + version you want.
These are real, verified-to-exist project URLs found via web search on
2026-09-08 (see dataset_report.md "Dataset Research" section for the search
sources); their live availability/version numbers can change, so check the
Universe page before running.
"""
from roboflow import Roboflow

# Fill in your free API key, or call rf = Roboflow() and use roboflow.login()
# interactively instead.
API_KEY = "YOUR_ROBOFLOW_API_KEY"

DATASETS = [
    # (workspace/project slug, version, format)
    {"dataset_url": "https://universe.roboflow.com/choi-t72ze/expiry-date/1", "format": "coco"},
    {"dataset_url": "https://universe.roboflow.com/coolstuff-rfnc5/expiry-date-detection-i3bav/1", "format": "coco"},
    {"dataset_url": "https://universe.roboflow.com/new-workspace-wfzw3/grocery-dataset-q9fj2/1", "format": "yolov8"},
]


def main():
    rf = Roboflow(api_key=API_KEY)
    for d in DATASETS:
        print(f"Downloading {d['dataset_url']} as {d['format']} ...")
        try:
            rf.download_dataset(dataset_url=d["dataset_url"], model_format=d["format"])
        except Exception as e:
            print(f"  FAILED ({e}). The project/version may have moved or been made "
                  f"private -- check the URL on universe.roboflow.com directly.")
    print("\nDone. Each dataset lands in its own folder named after the project.")
    print("Next: merge relevant images into legal_metrology_dataset/images/ under a new, "
          "clearly named subset (e.g. 'roboflow_expiry_date'), keep its own annotation "
          "format separate from ours, and re-run the product-level split logic in "
          "scripts/build_dataset.py before mixing into train/val/test.")


if __name__ == "__main__":
    main()
