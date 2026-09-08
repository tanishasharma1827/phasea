#!/usr/bin/env python3
"""
Recomputes reports/dataset_statistics.json and reports/annotation_quality_report.json
directly from the files actually on disk -- never from cached counters kept by
build_dataset.py / merge_openfoodfacts.py, which can drift out of date after
any merge. Safe to run any time, as many times as you like.

Usage:
    python scripts/recompute_reports.py --root .
"""
import argparse, os, csv, json
from collections import defaultdict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    args = ap.parse_args()
    root = args.root

    # ---- images.csv is the single source of truth for image/product counts ----
    images_csv = os.path.join(root, "metadata", "images.csv")
    rows = []
    with open(images_csv) as f:
        for row in csv.DictReader(f):
            rows.append(row)

    by_source = defaultdict(list)
    for r in rows:
        by_source[r["source_dataset"]].append(r)

    images_per_source = {src: len(rs) for src, rs in by_source.items()}
    products_per_source = {src: len(set(r["product_id"] for r in rs)) for src, rs in by_source.items()}
    splits_per_source = {
        src: {sp: sum(1 for r in rs if r["split"] == sp) for sp in ("train", "val", "test")}
        for src, rs in by_source.items()
    }
    countries_present = sorted(set(r.get("source_country", "UNKNOWN") for r in rows))

    # ---- OCR region counts, split by provenance, straight from the actual files ----
    def count_jsonl(path):
        if not os.path.isfile(path):
            return 0
        with open(path) as f:
            return sum(1 for _ in f)

    ocr_jsonl = os.path.join(root, "annotations", "ocr.jsonl")
    sroie_jsonl = os.path.join(root, "annotations", "existing", "sroie_receipt_ocr.jsonl")
    decl_jsonl = os.path.join(root, "annotations", "declarations.jsonl")
    prod_meta_jsonl = os.path.join(root, "annotations", "product_metadata_declarations.jsonl")

    # OCR regions by source (ocr.jsonl now mixes GroceryStoreDataset + OpenFoodFacts rows)
    ocr_by_source = defaultdict(int)
    if os.path.isfile(ocr_jsonl):
        with open(ocr_jsonl) as f:
            for line in f:
                rec = json.loads(line)
                ocr_by_source[rec.get("source", "UNKNOWN")] += 1

    decl_by_source_and_field = defaultdict(lambda: defaultdict(int))
    decl_field_totals = defaultdict(int)
    if os.path.isfile(decl_jsonl):
        with open(decl_jsonl) as f:
            for line in f:
                rec = json.loads(line)
                decl_by_source_and_field[rec.get("source", "UNKNOWN")][rec["field_type"]] += 1
                decl_field_totals[rec["field_type"]] += 1

    # quality metrics
    quality_csv = os.path.join(root, "annotations", "image_quality.csv")
    quality_counts = {"n_rows": 0, "blur_flag": 0, "low_light_flag": 0, "poor_contrast_flag": 0}
    if os.path.isfile(quality_csv):
        with open(quality_csv) as f:
            for row in csv.DictReader(f):
                quality_counts["n_rows"] += 1
                for k in ("blur_flag", "low_light_flag", "poor_contrast_flag"):
                    quality_counts[k] += int(row.get(k, 0) or 0)

    stats = {
        "_generated_by": "scripts/recompute_reports.py -- derived live from files on disk, not cached counters",
        "total_images_real": len(rows),
        "synthetic_images": 0,
        "images_per_source": images_per_source,
        "unique_products_per_source": products_per_source,
        "images_per_split_per_source": splits_per_source,
        "source_countries_present": countries_present,
        "ocr_text_regions_by_source_AUTO_or_ORIGINAL": dict(ocr_by_source),
        "sroie_ocr_text_regions_real_ORIGINAL_DATASET": count_jsonl(sroie_jsonl),
        "declaration_field_candidates_by_source": {k: dict(v) for k, v in decl_by_source_and_field.items()},
        "declaration_field_candidates_total": sum(decl_field_totals.values()),
        "declaration_field_totals_by_type": dict(decl_field_totals),
        "product_level_metadata_declarations_total": count_jsonl(prod_meta_jsonl),
        "image_quality_metrics_AUTO_CV": quality_counts,
        "mrp_candidates_found": decl_field_totals.get("MRP", 0),
        "note_mrp": ("MRP regex candidates require a real Indian Rupee / MRP-keyword context to fire; "
                      "0 here means either no Indian data has been merged yet, or none of the merged "
                      "images' OCR text matched the MRP pattern -- check manually before assuming failure."),
    }

    reports_dir = os.path.join(root, "reports")
    os.makedirs(reports_dir, exist_ok=True)
    with open(os.path.join(reports_dir, "dataset_statistics.json"), "w") as f:
        json.dump(stats, f, indent=2)

    # ---- annotation quality, recomputed the same way ----
    aq = {
        "_generated_by": "scripts/recompute_reports.py",
        "by_source": {},
    }
    for src in images_per_source:
        aq["by_source"][src] = {
            "images": images_per_source[src],
            "unique_products": products_per_source[src],
            "ocr_regions": ocr_by_source.get(src, 0),
            "declaration_field_candidates": sum(decl_by_source_and_field.get(src, {}).values()),
        }
    with open(os.path.join(reports_dir, "annotation_quality_report.json"), "w") as f:
        json.dump(aq, f, indent=2)

    print(json.dumps(stats, indent=2))
    print(f"\nWrote reports/dataset_statistics.json and reports/annotation_quality_report.json "
          f"(recomputed live from {len(rows)} rows in metadata/images.csv)")


if __name__ == "__main__":
    main()
