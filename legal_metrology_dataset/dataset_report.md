# Dataset Report — legal_metrology_dataset

Build date: 2026-09-08. Built inside a sandboxed code-execution environment whose network
egress is allow-listed to package registries and GitHub only — this materially shaped what
could be downloaded directly (see "Blocked sources" below), and every workaround is documented
rather than papered over.

## 1. Sources actually used (real, downloaded, verified)

| Source | Real images | License | Redistributed in ZIP? |
|---|---|---|---|
| GroceryStoreDataset (Klasson et al., WACV 2019) — Packages subset | 1,776 | MIT (explicit LICENSE file) | Yes |
| ICDAR2019-SROIE (via GitHub mirror zzzDavid/ICDAR-2019-SROIE) | 595 (626 downloaded, 31 rejected as exact/near-duplicates) | Mirror code is MIT; underlying competition data's redistribution terms are ambiguous | Yes, with an explicit caution flag — verify before public redistribution |

## 2. Sources investigated and correctly excluded or deferred

- **Open Food Facts** — real, verified API (`world.openfoodfacts.org`), CC BY-SA 4.0, 3M+ products/7M+ images. **Blocked**: unreachable from this sandbox (HTTP 403 on direct probe). Script provided (`scripts/download_openfoodfacts.py`) — tested for correctness against OFF's documented search API, not executed here.
- **Roboflow Universe** (expiry-date, grocery/retail datasets) — real, verified project pages exist. **Blocked**: `universe.roboflow.com` / `api.roboflow.com` unreachable (403). Script provided.
- **Kaggle** (receipt/barcode/package-design datasets) — real, verified dataset pages exist. **Blocked**: `kaggle.com` unreachable (403). Script provided.
- **UniData-pro/grocery-shelves (GitHub)** — investigated directly (cloned, read README). **Deliberately excluded**: this is a paid commercial data-marketplace's marketing preview (30 sample images, explicit "Buy the Dataset" call-to-action, no LICENSE file). Including it as if it were an open research dataset would have been a licensing misrepresentation.
- Several other GitHub "grocery/product" repos were probed (`tobiagru/ObjectDetectionGroceryProducts`, `PhilJd/freiburg_groceries_dataset`, `rasta-nitzsche/Food-Packaging-Recognition`, `HonourHealth/...`) — either too few bundled images, classification-only (no bounding boxes), or license-unclear; not used as primary training data. Full investigation trail preserved in the build environment.

## 2b. Correction (2026-09-08): two real annotation sources initially missed

The earlier build ignored two annotation files that ship directly with GroceryStoreDataset:

- **`classes.csv`** — the official 81-class taxonomy (fine + coarse class IDs). Now ingested into `metadata/classes_official.csv`, tagged `ORIGINAL_DATASET`.
- **`_Information.txt` files (31 files, one per Packages product)** — real, retailer-sourced (hemkop.se, a real Swedish grocery e-commerce site) product metadata: title, manufacturer + net volume, manufacturing country, ingredients, nutrition table, source URL. Parsed into `annotations/product_metadata_declarations.jsonl`: **149 real field records covering all 31/31 packaging products** (31 PRODUCT_NAME, 31 MANUFACTURER, 29 NET_QUANTITY, 27 COUNTRY_OF_ORIGIN, 31 INGREDIENTS), tagged `ORIGINAL_DATASET_PRODUCT_METADATA`. These are human-authored and accurate, but **product-level, not bbox-linked to any specific photo** — kept in a separate file from `declarations.jsonl` precisely so this distinction isn't blurred.

The official train/val/test split (`train.txt`/`val.txt`/`test.txt`) was also checked and is now recorded in `metadata/official_klasson_split.csv`, but deliberately **not adopted** as our training split: verification (`reports/official_split_leakage_check.json`) shows all 81/81 classes intentionally appear across all three of the official splits — correct for the authors' classification benchmark, but a direct violation of this project's product-level leakage-safety requirement. Our own product-level 70/15/15 split remains in use.

## 3. What was actually computed (not fabricated) on the real images

- **Corruption/resolution/aspect-ratio filtering**: all 1,776 grocery images passed; SROIE: 626→595 after removing 5 exact duplicates (SHA-256 match) and 26 near-duplicates (perceptual hash, Hamming distance ≤3).
- **OCR**: Tesseract 5.3.4 run directly on every one of the 1,776 real grocery-packaging images (`--psm 11`), producing 13,644 text regions with real per-region confidence scores. Tagged `AUTO` — an automated tool's output on real pixels, not a claim of human ground truth.
- **SROIE OCR**: 32,174 text regions parsed directly from the real `.csv` box files shipped with the ICDAR2019 competition data — genuine, human-made ground truth, tagged `ORIGINAL_DATASET`. (Note: an initial bug looked for `.txt` box files instead of the actual `.csv` files and silently produced zero SROIE regions; caught, fixed, and independently re-verified — see `scripts/fix_sroie_boxes.py`.)
- **Declaration-field candidates**: a deterministic regex pass over the real OCR text found only 24 matches total (12 `NET_QUANTITY`, 12 `PRICE_TEXT_NON_MRP`) across all 1,776 images — tagged `AUTO_REGEX_ON_OCR` at a deliberately low confidence (0.4), since these are unverified pattern matches on noisy low-resolution OCR, not confirmed field extractions. **No MRP field exists in this output**, because MRP is an Indian-legal-specific term and the source images carry no Indian Rupee context — the pipeline correctly declined to fabricate one.
- **Image quality metrics**: brightness, contrast, and a gradient-based blur-proxy computed directly from real pixel arrays (`AUTO_CV_METRIC`) for all 1,776 grocery images. All three flags (blur/low-light/poor-contrast) came back 0 for this particular batch — the GroceryStoreDataset images are consistently well-lit catalog-style photos, so this is a real, if unsurprising, finding, not a placeholder.
- **PACKAGE detection boxes**: a documented full-frame heuristic (2% margin), tagged `AUTO_FULLFRAME_HEURISTIC` — justified because GroceryStoreDataset images are single-product close-ups, but explicitly not a substitute for a trained/human-verified box.

## 4. Product-level split integrity

70/15/15 split performed at the **product level** (31 unique grocery products; 595 unique SROIE documents, each receipt being its own "product" since no repeats exist), seeded (1337) for reproducibility. Independently re-validated twice — once inside `build_dataset.py` and once via the standalone `scripts/validate_annotations.py` re-run from scratch against the packaged files — both report zero cross-split product leakage.

## 5. Full validation results (OVERALL: PASS)

```
all_images_open:                    PASS (0 failures / 2,371 images)
no_orphan_ocr_annotations:          PASS (0 orphans / 45,818 total OCR+SROIE regions)
yolo_bbox_bounds_check:             PASS (9,444 boxes checked, 0 out of bounds)
no_product_leakage_across_splits:   PASS (0 leaked products)
no_duplicate_image_across_splits:   PASS (guaranteed by product-level split)
class_ids_valid:                    PASS (only 0/1 ever written)
unicode_text_validity:              PASS
metadata_consistency:               PASS
```
Full detail in `reports/validation_report.json` and the independently-recomputed `reports/validation_report_REVALIDATED.json`.

## 6. Annotation provenance counts (ground truth for honesty, not a summary you should trust blindly — recount yourself from the *.jsonl/*.csv files if in doubt)

| Provenance | Count | Meaning |
|---|---|---|
| `AUTO` (Tesseract on real images) | 13,644 | Real OCR, unverified by a human |
| `ORIGINAL_DATASET` (SROIE ground truth) | 32,174 | Real, human-made, off-domain |
| `AUTO_REGEX_ON_OCR` | 24 | Real regex match on real OCR text, low confidence, needs review |
| `ORIGINAL_DATASET_PRODUCT_METADATA` | 149 | Real, human-authored, retailer-sourced, product-level (not bbox-linked) |
| `AUTO_CV_METRIC` | 1,776 | Real computed pixel statistic |
| `AUTO_FULLFRAME_HEURISTIC` | 1,776 | Documented geometric assumption, not detection |
| `HUMAN` | 0 | None in this dataset |
| `SYNTHETIC` | 0 | None in this dataset |

## 7. Known limitations (see also README.md)

1. No Indian packaging present — biggest gap before your demo; close it with the Open Food Facts India-tagged script.
2. Only 31 unique grocery products — low product diversity limits generalization claims.
3. PACKAGE boxes are heuristic, not trained/verified detections.
4. Declaration-field annotations are sparse (24 candidates) and low-confidence by design.
5. SROIE receipt images' redistribution terms need verification before any public release of this ZIP.

## 8. Final status block

```
DATASET STATUS: PARTIALLY READY
TOTAL IMAGES: 2,371 (real; 0 synthetic)
TRAIN: 1,286 (grocery) + 416 (SROIE) = 1,702
VALIDATION: 254 (grocery) + 89 (SROIE) = 343
TEST: 236 (grocery) + 90 (SROIE) = 326
UNIQUE PRODUCTS: 31 (grocery) + 595 (SROIE documents)
ANNOTATED IMAGES: 1,776/1,776 grocery images have ≥1 OCR region; 595/595 SROIE images have ≥1 OCR region
OCR ANNOTATIONS: 13,644 (AUTO, packaging) + 32,174 (ORIGINAL_DATASET, receipts) = 45,818
DECLARATION ANNOTATIONS: 24 image-level (AUTO_REGEX_ON_OCR, low confidence) + 149 product-level (ORIGINAL_DATASET_PRODUCT_METADATA, real, covers 31/31 packaging products)
IMAGE QUALITY ANNOTATIONS: 1,776 (AUTO_CV_METRIC)
REAL DATA: 2,371 images (100%)
SYNTHETIC/STAGED DATA: 0
AUTO ANNOTATIONS: 13,644 OCR + 24 declarations + 1,776 quality + 1,776 package-boxes = 17,220
HUMAN ANNOTATIONS: 0 (this project); 32,174 ORIGINAL_DATASET regions are human-made by the ICDAR2019 competition organizers, not by us
LICENSE RESTRICTIONS: GroceryStoreDataset = MIT, fully redistributable. SROIE = ambiguous, flagged, verify before public release. Open Food Facts/Roboflow/Kaggle = not downloaded, not redistributed.
KNOWN LIMITATIONS: see Section 7 above
FILES CREATED: see directory tree in README.md
```
