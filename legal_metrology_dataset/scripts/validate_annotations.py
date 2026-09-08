#!/usr/bin/env python3
"""
Standalone, independent re-validation of legal_metrology_dataset/.
Run this from inside the unzipped dataset root (or pass --root).
Does not trust any pre-computed report -- recomputes everything from the
actual files on disk, so it also catches corruption introduced by copying
or re-zipping the dataset.

Usage:
    cd legal_metrology_dataset
    python scripts/validate_annotations.py
"""
import argparse, os, json, csv
from collections import defaultdict
from PIL import Image


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    args = ap.parse_args()
    root = args.root

    results = {}

    # 1. every image_id in metadata/images.csv actually opens
    images_csv = os.path.join(root, "metadata", "images.csv")
    id_to_row = {}
    bad_opens = []
    with open(images_csv) as f:
        for row in csv.DictReader(f):
            id_to_row[row["image_id"]] = row
    # locate actual files (dynamically discover all image subsets under images/,
    # not a hardcoded list -- so newly merged subsets like indian_packaging are
    # found automatically without needing to edit this script every time)
    images_root = os.path.join(root, "images")
    search_dirs = [os.path.join(images_root, d) for d in os.listdir(images_root)] if os.path.isdir(images_root) else []
    id_to_path = {}
    for d in search_dirs:
        for dirpath, _, files in os.walk(d):
            for fn in files:
                stem = os.path.splitext(fn)[0]
                id_to_path[stem] = os.path.join(dirpath, fn)

    for iid in id_to_row:
        p = id_to_path.get(iid)
        if p is None:
            bad_opens.append((iid, "file_not_found"))
            continue
        try:
            Image.open(p).verify()
        except Exception as e:
            bad_opens.append((iid, f"corrupt:{e}"))
    results["all_images_open"] = {"status": "PASS" if not bad_opens else "FAIL",
                                   "failures": bad_opens[:50], "failure_count": len(bad_opens)}

    # 2. no orphan OCR annotations
    valid_ids = set(id_to_row.keys())
    orphans = 0
    ocr_path = os.path.join(root, "annotations", "ocr.jsonl")
    if os.path.isfile(ocr_path):
        with open(ocr_path) as f:
            for line in f:
                rec = json.loads(line)
                if rec["image_id"] not in valid_ids:
                    orphans += 1
    results["no_orphan_ocr_annotations"] = {"status": "PASS" if orphans == 0 else "FAIL",
                                             "orphan_count": orphans}

    # 3. YOLO bbox bounds check
    lbl_dir = os.path.join(root, "annotations", "yolo", "labels")
    checked, bad = 0, 0
    if os.path.isdir(lbl_dir):
        for fn in os.listdir(lbl_dir):
            with open(os.path.join(lbl_dir, fn)) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) != 5:
                        continue
                    checked += 1
                    _, cx, cy, bw, bh = parts
                    cx, cy, bw, bh = float(cx), float(cy), float(bw), float(bh)
                    if not (0 <= cx <= 1 and 0 <= cy <= 1 and 0 < bw <= 1 and 0 < bh <= 1):
                        bad += 1
    results["yolo_bbox_bounds_check"] = {"status": "PASS" if bad == 0 else "FAIL",
                                          "boxes_checked": checked, "boxes_out_of_bounds": bad}

    # 4. product-level leakage across splits
    prod_splits = defaultdict(set)
    for row in id_to_row.values():
        prod_splits[row["product_id"]].add(row["split"])
    leaked = {pid: list(s) for pid, s in prod_splits.items() if len(s) > 1}
    results["no_product_leakage_across_splits"] = {"status": "PASS" if not leaked else "FAIL",
                                                     "leaked_products": leaked}

    # 5. class ids valid (only 0/1 for this dataset's PACKAGE/TEXT_REGION scheme)
    bad_class = 0
    if os.path.isdir(lbl_dir):
        for fn in os.listdir(lbl_dir):
            with open(os.path.join(lbl_dir, fn)) as f:
                for line in f:
                    parts = line.strip().split()
                    if not parts:
                        continue
                    if parts[0] not in ("0", "1"):
                        bad_class += 1
    results["class_ids_valid"] = {"status": "PASS" if bad_class == 0 else "FAIL", "bad_lines": bad_class}

    overall = "PASS" if all(v.get("status") == "PASS" for v in results.values()) else "FAIL"
    results["OVERALL"] = overall

    print(json.dumps(results, indent=2, default=str))
    out_path = os.path.join(root, "reports", "validation_report_REVALIDATED.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
