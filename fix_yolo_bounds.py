#!/usr/bin/env python3
"""
Fixes out-of-bounds YOLO TEXT_REGION boxes caused by a real bug in an earlier
version of merge_openfoodfacts.py: PaddleOCR boxes were converted to YOLO's
normalized [0,1] format WITHOUT first clamping them to the image's actual
width/height, so a handful of boxes (typically from internal PaddleOCR
preprocessing/rounding) ended up slightly outside the image.

This does NOT re-run OCR -- the raw text/bbox data in annotations/ocr.jsonl
is already correct; only the YOLO conversion step was wrong. This script
regenerates every annotations/yolo/labels/<image_id>.txt file from
ocr.jsonl + metadata/images.csv, with correct clamping this time, for ALL
images (grocery + SROIE-off-domain-excluded + any merged Indian subset),
so the whole dataset is consistent after running this once.

Usage:
    python scripts/fix_yolo_bounds.py --root .
"""
import argparse, os, csv, json
from collections import defaultdict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    args = ap.parse_args()
    root = args.root

    # image_id -> (width, height) from metadata/images.csv (ground truth dims)
    dims = {}
    with open(os.path.join(root, "metadata", "images.csv")) as f:
        for row in csv.DictReader(f):
            dims[row["image_id"]] = (int(row["width"]), int(row["height"]))

    # group OCR regions by image_id
    ocr_by_image = defaultdict(list)
    ocr_path = os.path.join(root, "annotations", "ocr.jsonl")
    with open(ocr_path) as f:
        for line in f:
            rec = json.loads(line)
            ocr_by_image[rec["image_id"]].append(rec)

    lbl_dir = os.path.join(root, "annotations", "yolo", "labels")
    fixed = 0
    dropped_boxes = 0
    total_boxes = 0

    for image_id, (w, h) in dims.items():
        if image_id not in ocr_by_image and not os.path.isfile(os.path.join(lbl_dir, image_id + ".txt")):
            continue  # nothing to write for this image (e.g. SROIE, which isn't in yolo/ at all)
        lines = ["0 0.500000 0.500000 0.960000 0.960000"]  # PACKAGE heuristic, unchanged
        for reg in ocr_by_image.get(image_id, []):
            if reg["ocr_confidence"] < 0.30:
                continue
            bx1, by1, bx2, by2 = reg["bbox"]
            bx1, bx2 = max(0, bx1), min(w, bx2)
            by1, by2 = max(0, by1), min(h, by2)
            total_boxes += 1
            if bx2 <= bx1 or by2 <= by1:
                dropped_boxes += 1
                continue
            ccx, ccy = (bx1 + bx2) / 2 / w, (by1 + by2) / 2 / h
            cbw, cbh = (bx2 - bx1) / w, (by2 - by1) / h
            # final safety clamp against float rounding at the normalized-coordinate level too
            ccx, ccy = min(max(ccx, 0.0), 1.0), min(max(ccy, 0.0), 1.0)
            cbw, cbh = min(max(cbw, 1e-6), 1.0), min(max(cbh, 1e-6), 1.0)
            lines.append(f"1 {ccx:.6f} {ccy:.6f} {cbw:.6f} {cbh:.6f}")
        lbl_path = os.path.join(lbl_dir, image_id + ".txt")
        if os.path.isfile(lbl_path):  # only rewrite images that were already part of the yolo/ set
            with open(lbl_path, "w") as f:
                f.write("\n".join(lines) + "\n")
            fixed += 1

    print(f"Rewrote {fixed} label files.")
    print(f"Text-region boxes processed: {total_boxes}, dropped as degenerate after clamping: {dropped_boxes}")
    print("Run scripts/validate_annotations.py next to confirm yolo_bbox_bounds_check now PASSes.")


if __name__ == "__main__":
    main()
