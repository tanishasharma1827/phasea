# legal_metrology_dataset — READ THIS FIRST

**Status: PARTIALLY READY.** This is a real, honestly-labelled dataset built entirely from
publicly available, license-checked sources. It is **not** a Legal Metrology compliance
dataset by itself — no such dataset exists publicly (verified). It is a **foundation** for
Phase C (Models 1, 2, 4) plus a smaller, honestly-labelled starting point for Model 3, with
clear next steps to close the remaining gaps before your demo.

**Correction note (2026-09-08):** an earlier pass of this dataset failed to ingest two real,
already-existing annotation sources shipped with GroceryStoreDataset: the official
`classes.csv` taxonomy and, more importantly, 31 real `_Information.txt` files (retailer-
sourced product metadata: manufacturer, net volume, country of origin, ingredients — see
`annotations/product_metadata_declarations.jsonl`, 149 real field records). Both are now
correctly ingested and tagged `ORIGINAL_DATASET` / `ORIGINAL_DATASET_PRODUCT_METADATA`. The
official train/val/test split was also checked and deliberately **not** adopted — it's a
per-image classification split where all 81 classes intentionally appear in every split
(see `reports/official_split_leakage_check.json`), which would violate this project's
product-level leakage-safety requirement.

## What is actually real vs. what still needs you

| | Status |
|---|---|
| 2,371 real images (1,776 packaging + 595 receipts) | ✅ Real, downloaded, cleaned, deduplicated |
| Product-level train/val/test split, leakage-checked | ✅ Real, verified PASS (see `reports/leakage_report.json`) |
| 13,644 OCR text regions on packaging images | ✅ Real Tesseract 5.3.4 output on real pixels (`AUTO`) |
| 32,174 OCR text regions on receipts | ✅ Real, human-made ICDAR2019 competition ground truth (`ORIGINAL_DATASET`) |
| PACKAGE detection boxes | ⚠️ `AUTO_FULLFRAME_HEURISTIC` — a documented full-frame assumption, not a trained detector's output. Fine as a Model-1 warm-start target since these are single-product close-ups, but **not equivalent to human-verified boxes**. |
| TEXT_REGION detection boxes | ✅ Real Tesseract boxes, confidence-filtered ≥0.30 |
| Legal Metrology declaration fields (MRP/net qty/manufacturer/etc.), image-level, from OCR | ⚠️ Only 24 low-confidence candidate hits (12 NET_QUANTITY, 12 non-INR price text), tagged `AUTO_REGEX_ON_OCR`, confidence 0.4 — these are unverified regex matches on noisy OCR of low-res catalog photos. |
| Declaration-like fields, product-level, from real catalog metadata | ✅ **149 real records across all 31 packaging products** (PRODUCT_NAME, MANUFACTURER, NET_QUANTITY, COUNTRY_OF_ORIGIN, INGREDIENTS) — retailer-sourced (hemkop.se), human-authored, tagged `ORIGINAL_DATASET_PRODUCT_METADATA`. See `annotations/product_metadata_declarations.jsonl`. **Important caveat:** these apply to the *product class*, not to a specific bounding box in a specific photo — there is no pixel-level link between this text and where it appears on any given package image. Use it to train/validate Model 3's text-classification behavior (e.g. "does this string look like a manufacturer name"), not as a substitute for photographed, bbox-linked declarations. |
| Indian packaging / Indian products | ❌ **None.** The packaging subset is Swedish grocery products (dairy/juice). Explicitly tagged `NON_INDIAN` everywhere. See Part A2 gap below. |
| Open Food Facts, Roboflow, Kaggle data | ❌ Not downloaded — network-blocked from this build environment. Working scripts provided for you to run locally. |

## Directory structure

```
legal_metrology_dataset/
├── images/international_packaging/{train,val,test}/   1,776 real Swedish grocery-packaging photos (MIT license)
├── images/offdomain_ocr_practice/{train,val,test}/     595 real Malaysian receipt photos (off-domain OCR practice)
├── annotations/
│   ├── ocr.jsonl                  13,644 real AUTO Tesseract regions (packaging)
│   ├── existing/sroie_receipt_ocr.jsonl   32,174 real ORIGINAL_DATASET regions (receipts)
│   ├── declarations.jsonl         24 low-confidence AUTO_REGEX_ON_OCR candidates (packaging, image-level)
│   ├── product_metadata_declarations.jsonl  149 real ORIGINAL_DATASET_PRODUCT_METADATA records (product-level, not bbox-linked)
│   ├── image_quality.csv          1,776 rows, real AUTO_CV_METRIC blur/brightness/contrast
│   └── yolo/                      YOLO-format detection labels (class 0 PACKAGE, class 1 TEXT_REGION)
├── metadata/                      images.csv, products.csv, splits.csv, provenance.csv,
│                                   classes_official.csv (real 81-class taxonomy),
│                                   official_klasson_split.csv (recorded, not used — see below)
├── splits/                        train.txt / val.txt / test.txt (image_id lists)
├── configs/                       classes.txt, dataset.yaml (Ultralytics-ready)
├── reports/                       dataset_statistics.json, cleaning/leakage/validation/annotation-quality reports
├── external_sources/              scripts + manifest for the 3 sources this sandbox couldn't reach
├── scripts/                       build_dataset.py, fix_sroie_boxes.py, validate_annotations.py, download_*.py
└── source_manifest.csv            top-level copy of metadata/provenance.csv
```

## How to use this for Phase C

1. **Model 1 (package/label detector):** train YOLO26 on `annotations/yolo/` using `configs/dataset.yaml`, filtering images by `metadata/splits.csv`. Treat PACKAGE-box accuracy as a warm-start, not ground truth — the box is a full-frame heuristic (see limitations).
2. **Model 2 (OCR):** the packaging images are low-resolution (min side 198px, typical 348×348) — before fine-tuning, benchmark PaddleOCR's out-of-box CER/WER against `annotations/ocr.jsonl`'s Tesseract baseline on the same images, since PaddleOCR is likely to outperform Tesseract on small text.
3. **Model 3 (field classifier):** the 24 image-level candidate declarations + 149 real product-level records here are a start, but not enough on their own. **Closing this gap is now a two-command process** — see "Adding real Indian data" below.
4. **Model 4 (image quality):** `annotations/image_quality.csv` gives you a real, computed baseline (blur variance, brightness, contrast) on all 1,776 packaging images — usable as-is for a rule-based or lightly-trained readability score.

## Adding real Indian data (tested, not just described)

This sandbox cannot reach `world.openfoodfacts.org` (network-blocked), so no real Indian images
are in this ZIP. Two scripts close that gap — run both **on your own machine**, in order:

```bash
python scripts/download_openfoodfacts.py --country india --limit 500 --out ./off_download
python scripts/merge_openfoodfacts.py --off-dir ./off_download --dataset-root ./legal_metrology_dataset
```

What `merge_openfoodfacts.py` actually does, in order: cleans/deduplicates the new images
(including against everything already in this dataset, so it can't re-add something you already
have); assigns a fresh product-level 70/15/15 split **only to genuinely new products**, leaving
every existing split assignment untouched; copies images into a new, honestly-labelled
`images/indian_packaging/{train,val,test}/` subset (never mixed with the Swedish
`international_packaging` subset); runs the same real Tesseract OCR used elsewhere in this
dataset, **plus an MRP/₹/FSSAI regex pattern that was deliberately absent before** (there was no
Indian Rupee context to match against until now — still tagged `AUTO_REGEX_ON_OCR`, still
confidence 0.4, still needs human review, but now actually fires); and appends to every metadata
file without overwriting existing rows. It finishes by re-running the leakage check across the
**combined** dataset.

**This merge script was tested** against a synthetic stand-in fixture (12 real image files reused
as fake "Open Food Facts" rows, purely to exercise the code path — never presented as real OFF
data) on a throwaway copy of this dataset, not against your actual delivered files. That test
caught and fixed one real bug: `validate_annotations.py` had a hardcoded list of image
subdirectories and didn't know to look in a newly-added `indian_packaging` folder, so it falsely
reported "file not found" even though the images were present and opened fine. Fixed to discover
subdirectories dynamically — already applied in the `scripts/` folder in this ZIP.

After merging, always run `python scripts/validate_annotations.py --root .` yourself before
training — it independently re-verifies everything from the files on disk, not from a report
that might be stale.



## Before you redistribute or submit this publicly

- The SROIE/receipt subset's underlying image redistribution rights are **not unambiguous** (see `source_manifest.csv` note) — verify against the official ICDAR Robust Reading Competition terms before including it in a public GitHub repo or public submission. It is safe to keep for internal training/practice.
- The GroceryStoreDataset (Klasson et al.) subset carries an explicit MIT license and is safe to redistribute with attribution.

## Known limitations (stated plainly, not buried)

1. **No Indian packaging images are in this dataset.** Everything real here is Swedish (packaging) or Malaysian (receipts). Your MRP/consumer-care/country-of-origin field examples will need to come from Open Food Facts' India-tagged subset (script provided) or your own captures.
2. **Only 31 unique products** in the packaging subset (avg. ~57 images/product) — the product-level split is leakage-safe but the product *diversity* is low. Don't expect strong generalization claims from this subset alone; treat it as pretraining, not final training data.
3. **PACKAGE boxes are a documented heuristic**, not detections — full-frame minus a 2% margin. This is a reasonable Model-1 warm-start given these are single-product close-ups, but say so explicitly in your evaluation report; do not present it as human-verified.
4. **Image-level, bbox-linked declaration coverage is thin (24 OCR-derived candidates across 1,776 images)** because the source images are low-resolution product-catalog shots, not full-label close-ups, and use non-Indian currency/formats. Product-level metadata (149 records, item 2 above) partly compensates for Model 3's text-classification training, but does not substitute for photographed, localizable declarations — closing this with real Indian packaging photos (via Open Food Facts) is still the highest-priority next step.
5. **The official Klasson train/val/test split is not used for training** — recorded in `metadata/official_klasson_split.csv` for reference, but our own product-level split is what's actually used, for the leakage-safety reason explained above.
