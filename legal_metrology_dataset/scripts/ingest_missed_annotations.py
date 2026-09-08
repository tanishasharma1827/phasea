#!/usr/bin/env python3
"""
CORRECTION SCRIPT — ingests two real, human-authored annotation sources that
the original build_dataset.py failed to load:

  1. dataset/classes.csv — the official Klasson et al. fine/coarse class
     taxonomy (81 classes; product_id/class_id/coarse_class_id).
  2. dataset/iconic-images-and-descriptions/Packages/**/*_Information.txt —
     real retailer-sourced product metadata (manufacturer, net volume,
     manufacturing country, ingredients, nutrition table), scraped by the
     dataset authors from hemkop.se (a real Swedish grocery retailer) for
     each of the 31 Packages products.

These are genuine ORIGINAL_DATASET-provenance annotations (human/editorially
authored, not something we generated) -- but they are PRODUCT-level metadata,
not per-image pixel annotations: there is no bounding box tying "Manufacturer
and volume: ARLA KO , 1l" to a specific region of a specific photo, so every
record here has bbox=null and is tagged annotation_method=ORIGINAL_DATASET_
PRODUCT_METADATA to keep it clearly distinct from the AUTO/Tesseract,
per-image, bbox-linked OCR regions already in ocr.jsonl.

We also did NOT switch the train/val/test split to the official Klasson
split, because that split is a per-image CLASSIFICATION split (every one of
the 81 classes intentionally appears in all three of train/val/test -- see
official_split_leakage_check.json) which is correct for their benchmark task
but violates this project's leakage-safety requirement. We keep our own
product-level split and additionally RECORD the official split per image for
transparency (metadata/official_klasson_split.csv).
"""
import os, csv, json, re

GSD = "/home/claude/probe/marcusklasson_GroceryStoreDataset/dataset"
OUT = "/home/claude/dsbuild/legal_metrology_dataset"

# ---------------------------------------------------------------------------
# 1. classes.csv -> metadata/classes_official.csv
# ---------------------------------------------------------------------------
classes_out = os.path.join(OUT, "metadata", "classes_official.csv")
n_classes = 0
with open(os.path.join(GSD, "classes.csv"), encoding="utf-8-sig") as f_in, \
     open(classes_out, "w", newline="", encoding="utf-8") as f_out:
    reader = csv.reader(f_in)
    writer = csv.writer(f_out)
    header = next(reader)
    writer.writerow(["class_name", "class_id", "coarse_class_name", "coarse_class_id",
                      "iconic_image_path", "product_description_path", "source", "annotation_method"])
    for row in reader:
        if len(row) < 6:
            continue
        writer.writerow(row[:6] + ["GroceryStoreDataset_Klasson2019", "ORIGINAL_DATASET"])
        n_classes += 1
print(f"Wrote metadata/classes_official.csv ({n_classes} real official classes)")

# ---------------------------------------------------------------------------
# 2. Official train/val/test split -> metadata/official_klasson_split.csv
#    (recorded for transparency; NOT used as our training split -- see docstring)
# ---------------------------------------------------------------------------
from collections import defaultdict
official_rows = []
prod_official_splits = defaultdict(set)
for split, fn in [("train", "train.txt"), ("val", "val.txt"), ("test", "test.txt")]:
    path = os.path.join(GSD, fn)
    if not os.path.isfile(path):
        continue
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            img_path, fine_id, coarse_id = [x.strip() for x in line.split(",")]
            product = img_path.split("/")[-2]
            official_rows.append({"relative_path": img_path, "product_id": product,
                                   "official_split": split, "fine_class_id": fine_id,
                                   "coarse_class_id": coarse_id})
            prod_official_splits[product].add(split)

leaked = {p: sorted(s) for p, s in prod_official_splits.items() if len(s) > 1}
with open(os.path.join(OUT, "reports", "official_split_leakage_check.json"), "w") as f:
    json.dump({
        "note": "This checks the ORIGINAL Klasson et al. train/val/test split, not ours. "
                "It is a per-image classification split where every class intentionally "
                "appears in all three splits -- correct for their benchmark, but why we did "
                "NOT use it as our leakage-safe product-level split.",
        "total_products_in_official_index": len(prod_official_splits),
        "products_appearing_in_multiple_official_splits": len(leaked),
        "conclusion": "Official split is NOT product-level leakage-safe by design (intentional "
                      "for classification benchmarking). Our own product-level 70/15/15 split "
                      "(seed=1337, see build_dataset.py) remains the correct choice for this "
                      "project's detector/field-extraction use case.",
    }, f, indent=2)

with open(os.path.join(OUT, "metadata", "official_klasson_split.csv"), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["relative_path", "product_id", "official_split", "fine_class_id", "coarse_class_id"])
    for r in official_rows:
        w.writerow([r["relative_path"], r["product_id"], r["official_split"],
                    r["fine_class_id"], r["coarse_class_id"]])
print(f"Wrote metadata/official_klasson_split.csv ({len(official_rows)} rows) "
      f"+ reports/official_split_leakage_check.json "
      f"({len(leaked)}/{len(prod_official_splits)} products span multiple official splits)")

# ---------------------------------------------------------------------------
# 3. Parse real _Information.txt files for the 31 Packages products
# ---------------------------------------------------------------------------
FIELD_LINE_RE = {
    "product_name": re.compile(r"^Title:\s*(.+)$"),
    "manufacturer_and_volume": re.compile(r"^Manufacturer and volume:\s*(.+)$"),
    "country_of_origin": re.compile(r"^Manufacturing country:\s*(.+)$"),
    "ingredients": re.compile(r"^Ingredients:\s*(.+)$"),
    "source_url": re.compile(r"^URL:\s*(.+)$"),
}
MANU_VOL_RE = re.compile(r"^(.*?)\s*,\s*([\d.,]+\s?(?:l|ml|kg|g|cl))\s*$", re.IGNORECASE)

pkg_root = os.path.join(GSD, "iconic-images-and-descriptions", "Packages")
records = []
n_files = 0
for coarse in sorted(os.listdir(pkg_root)) if os.path.isdir(pkg_root) else []:
    coarse_dir = os.path.join(pkg_root, coarse)
    if not os.path.isdir(coarse_dir):
        continue
    for fine in sorted(os.listdir(coarse_dir)):
        fine_dir = os.path.join(coarse_dir, fine)
        if not os.path.isdir(fine_dir):
            continue
        info_path = os.path.join(fine_dir, f"{fine}_Information.txt")
        if not os.path.isfile(info_path):
            continue
        n_files += 1
        with open(info_path, encoding="utf-8", errors="replace") as f:
            text = f.read()
        fields = {}
        for key, pattern in FIELD_LINE_RE.items():
            m = pattern.search(text, re.MULTILINE)
            if not m:
                # search line by line since URL sometimes glued to last nutrition line
                for line in text.splitlines():
                    mm = pattern.match(line.strip())
                    if mm:
                        fields[key] = mm.group(1).strip()
                        break
            else:
                fields[key] = m.group(1).strip()
        manufacturer, net_quantity = None, None
        if "manufacturer_and_volume" in fields:
            mm = MANU_VOL_RE.match(fields["manufacturer_and_volume"])
            if mm:
                manufacturer, net_quantity = mm.group(1).strip(), mm.group(2).strip()
            else:
                manufacturer = fields["manufacturer_and_volume"]
        record = {
            "product_id": fine,
            "coarse_category": coarse,
            "product_name": fields.get("product_name"),
            "manufacturer": manufacturer,
            "net_quantity": net_quantity,
            "country_of_origin": fields.get("country_of_origin"),
            "ingredients_text": fields.get("ingredients"),
            "source_url": fields.get("source_url"),
            "source": "GroceryStoreDataset_Klasson2019_ProductInformation",
            "annotation_method": "ORIGINAL_DATASET_PRODUCT_METADATA",
            "note": "Real retailer-sourced (hemkop.se) product metadata, applies to the whole "
                    "product class, not bbox-linked to any single photo.",
        }
        records.append(record)

print(f"\nParsed {n_files} real _Information.txt files -> {len(records)} product-level records")

# ---------------------------------------------------------------------------
# 4. Write as new declaration records into a SEPARATE file (kept apart from
#    declarations.jsonl's per-image, bbox-linked AUTO_REGEX_ON_OCR records,
#    per the note above about product-level vs image-level annotation).
# ---------------------------------------------------------------------------
product_decl_path = os.path.join(OUT, "annotations", "product_metadata_declarations.jsonl")
field_counts = defaultdict(int)
with open(product_decl_path, "w") as f:
    for r in records:
        for field_type, key in [("PRODUCT_NAME", "product_name"), ("MANUFACTURER", "manufacturer"),
                                 ("NET_QUANTITY", "net_quantity"), ("COUNTRY_OF_ORIGIN", "country_of_origin"),
                                 ("INGREDIENTS", "ingredients_text")]:
            val = r.get(key)
            if not val:
                continue
            f.write(json.dumps({
                "product_id": r["product_id"],
                "coarse_category": r["coarse_category"],
                "field_type": field_type,
                "text": val,
                "bbox": None,
                "source": r["source"],
                "source_url": r["source_url"],
                "annotation_method": "ORIGINAL_DATASET_PRODUCT_METADATA",
                "confidence": 1.0,
                "scope": "product_level_not_image_level",
            }) + "\n")
            field_counts[field_type] += 1

print(f"Wrote annotations/product_metadata_declarations.jsonl")
for k, v in field_counts.items():
    print(f"  {k}: {v}")

# ---------------------------------------------------------------------------
# 5. Update dataset_statistics.json / annotation_quality_report.json honestly
# ---------------------------------------------------------------------------
stats_path = os.path.join(OUT, "reports", "dataset_statistics.json")
with open(stats_path) as f:
    stats = json.load(f)
stats["real_official_classes_ingested"] = n_classes
stats["product_level_metadata_declarations"] = dict(field_counts)
stats["product_level_metadata_declarations_total"] = sum(field_counts.values())
stats["product_level_metadata_source"] = ("31 real _Information.txt files (real retailer-sourced "
    "product metadata: manufacturer, net volume, country of origin, ingredients) -- previously "
    "NOT ingested. Corrected 2026-09-08.")
stats["annotation_provenance_counts"]["ORIGINAL_DATASET_PRODUCT_METADATA"] = sum(field_counts.values())
stats["official_klasson_split_note"] = ("Official split recorded in metadata/official_klasson_split.csv "
    "but NOT used for training -- it is a per-image classification split, not product-level "
    "leakage-safe (see reports/official_split_leakage_check.json).")
with open(stats_path, "w") as f:
    json.dump(stats, f, indent=2)

aq_path = os.path.join(OUT, "reports", "annotation_quality_report.json")
with open(aq_path) as f:
    aq = json.load(f)
aq["by_source"]["GroceryStoreDataset_Klasson2019_ProductInformation"] = {
    "products_covered": len(records),
    "field_records": sum(field_counts.values()),
    "field_breakdown": dict(field_counts),
    "method": "ORIGINAL_DATASET_PRODUCT_METADATA (real, retailer-sourced, product-level, not bbox-linked)",
    "human_annotations": sum(field_counts.values()),
    "correction_note": "This source was NOT ingested in the original build; added 2026-09-08 after "
                        "user flagged that classes.csv/_Information.txt were real annotations being ignored.",
}
with open(aq_path, "w") as f:
    json.dump(aq, f, indent=2)

print("\nPatched dataset_statistics.json and annotation_quality_report.json.")
print("\nDONE.")
