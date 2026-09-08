#!/usr/bin/env python3
"""
Patch: the original build_dataset.py looked for SROIE box files as <stem>.txt,
but the real files on disk are <stem>.csv. This produced zero SROIE OCR regions
(verified: existing/sroie_receipt_ocr.jsonl was 0 bytes). This script re-derives
the SROIE image_id -> stem mapping from the already-written metadata/images.csv
(no re-download, no re-OCR of the grocery images needed) and correctly parses
the .csv box files, which are real, human-made ICDAR2019 competition ground truth.
"""
import os, csv, json, glob

ROOT = "/home/claude/dsbuild"
OUT = os.path.join(ROOT, "legal_metrology_dataset")
SROIE_BOX_DIR = "/home/claude/probe/sroie1/data/box"

# Rebuild image_id -> (product_id/stem) for SROIE rows from images.csv
sroie_rows = []
with open(os.path.join(OUT, "metadata", "images.csv")) as f:
    for row in csv.DictReader(f):
        if row["source_dataset"] == "ICDAR2019_SROIE":
            sroie_rows.append(row)

print(f"SROIE image rows found in images.csv: {len(sroie_rows)}")

out_path = os.path.join(OUT, "annotations", "existing", "sroie_receipt_ocr.jsonl")
count = 0
docs_with_boxes = 0
missing_box_file = 0

with open(out_path, "w") as out_f:
    for row in sroie_rows:
        stem = row["product_id"]
        box_path = os.path.join(SROIE_BOX_DIR, stem + ".csv")
        if not os.path.isfile(box_path):
            missing_box_file += 1
            continue
        had_any = False
        with open(box_path, encoding="utf-8", errors="replace") as bf:
            for line in bf:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(",")
                if len(parts) < 9:
                    continue
                try:
                    coords = list(map(int, parts[:8]))
                except ValueError:
                    continue
                text = ",".join(parts[8:])
                xs = coords[0::2]
                ys = coords[1::2]
                bbox = [min(xs), min(ys), max(xs), max(ys)]
                out_f.write(json.dumps({
                    "image_id": row["image_id"],
                    "document_id": stem,
                    "bbox": bbox,
                    "original_text": text,
                    "normalized_text": text.strip().lower(),
                    "source": "ICDAR2019_SROIE",
                    "annotation_method": "ORIGINAL_DATASET",
                    "domain": "receipt_OFFDOMAIN_not_packaging",
                }) + "\n")
                count += 1
                had_any = True
        if had_any:
            docs_with_boxes += 1

print(f"SROIE real ORIGINAL_DATASET text regions written: {count}")
print(f"SROIE receipts with at least one box: {docs_with_boxes}")
print(f"SROIE image rows with missing .csv box file: {missing_box_file}")

# ---- patch dataset_statistics.json ----
stats_path = os.path.join(OUT, "reports", "dataset_statistics.json")
with open(stats_path) as f:
    stats = json.load(f)
stats["sroie_ocr_text_regions_real_ORIGINAL_DATASET"] = count
stats["annotation_provenance_counts"]["ORIGINAL_DATASET (SROIE human-made ground truth)"] = count
stats["_patch_note"] = ("Fixed 2026-09-08: original script looked for SROIE box files as .txt; "
                         "real files are .csv. Re-parsed correctly, no re-download or re-OCR needed.")
with open(stats_path, "w") as f:
    json.dump(stats, f, indent=2)

# ---- patch annotation_quality_report.json ----
aq_path = os.path.join(OUT, "reports", "annotation_quality_report.json")
with open(aq_path) as f:
    aq = json.load(f)
aq["by_source"]["ICDAR2019_SROIE"]["ocr_regions"] = count
aq["by_source"]["ICDAR2019_SROIE"]["human_annotations"] = count
with open(aq_path, "w") as f:
    json.dump(aq, f, indent=2)

print("\nPatched dataset_statistics.json and annotation_quality_report.json.")

# ---- validation: no orphan sroie ocr annotations ----
valid_ids = set(r["image_id"] for r in sroie_rows)
orphans = 0
with open(out_path) as f:
    for line in f:
        rec = json.loads(line)
        if rec["image_id"] not in valid_ids:
            orphans += 1
print(f"Orphan check on SROIE OCR file: {orphans} orphans (should be 0)")

val_path = os.path.join(OUT, "reports", "validation_report.json")
with open(val_path) as f:
    val = json.load(f)
val["sroie_ocr_orphan_check_after_fix"] = {"status": "PASS" if orphans == 0 else "FAIL", "orphan_count": orphans}
with open(val_path, "w") as f:
    json.dump(val, f, indent=2)
print("Patched validation_report.json with post-fix check.")
