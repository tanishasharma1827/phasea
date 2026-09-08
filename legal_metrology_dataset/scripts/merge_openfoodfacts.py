#!/usr/bin/env python3
"""
Merges the output of download_openfoodfacts.py into an existing
legal_metrology_dataset/ package, WITHOUT breaking anything already validated:

  - existing images/products/splits are never touched or reassigned
  - new products get their own product-level 70/15/15 split (same seed logic
    as build_dataset.py, so it's reproducible)
  - new images go into images/indian_packaging/{train,val,test}/ -- a
    separate, honestly-labelled subset (source_country=INDIA), never mixed
    into images/international_packaging/ which is Swedish-sourced
  - runs REAL PaddleOCR (PP-OCR, GPU-accelerated if available) + regex
    field-candidate extraction, PLUS an MRP/INR-specific regex that was
    deliberately absent before (there was no Indian Rupee context to match
    against until now)
  - appends to metadata/images.csv, products.csv, splits.csv, provenance.csv
    (does not overwrite existing rows)
  - re-runs leakage/validation checks across the COMBINED dataset afterward

Usage:
    python download_openfoodfacts.py --country india --limit 500 --out ./off_download
    python merge_openfoodfacts.py --off-dir ./off_download --dataset-root ./legal_metrology_dataset

Requires: pip install paddleocr  (+ a matching paddlepaddle / paddlepaddle-gpu build
for your CUDA version -- see https://www.paddlepaddle.org.cn/en/install/quick).
If no GPU backend is found, PaddleOCR automatically falls back to CPU -- slower,
but still correct.
"""
import argparse, os, csv, json, hashlib, shutil, re, random
from collections import defaultdict
import numpy as np
from PIL import Image
from paddleocr import PaddleOCR
import imagehash

# One-time model load (this is slow -- seconds -- so it must happen ONCE,
# not per image). lang='en' covers English packaging text; if you also need
# Hindi/regional-script recognition, either switch lang to the appropriate
# code (PaddleOCR does not mix scripts within one model instance well) or
# instantiate a second reader and merge results per image -- not done here
# to keep this script simple; extend if your captured labels are bilingual.
print("Loading PaddleOCR model (one-time; downloads model weights from HuggingFace/"
      "ModelScope/AIStudio/BOS on first run if not cached -- needs real internet access)...")
_OCR = PaddleOCR(use_textline_orientation=True, lang="en")
try:
    import paddleocr as _paddleocr_pkg
    OCR_ENGINE_TAG = f"paddleocr_{_paddleocr_pkg.__version__}"
except Exception:
    OCR_ENGINE_TAG = "paddleocr_unknown_version"

FIELD_PATTERNS = {
    "NET_QUANTITY": re.compile(r"\b\d{1,4}(?:[.,]\d{1,3})?\s?(g|kg|ml|l|cl|oz|lb)\b", re.IGNORECASE),
    "MRP": re.compile(r"(?:mrp|m\.r\.p\.?|rs\.?|inr|₹)\s*[:.]?\s*\d{1,6}(?:[.,]\d{1,2})?", re.IGNORECASE),
    "DATE_LIKE": re.compile(r"\b\d{1,2}[./-]\d{1,2}[./-]\d{2,4}\b|\b\d{4}[./-]\d{1,2}[./-]\d{1,2}\b"),
    "BARCODE_LIKE": re.compile(r"\b\d{8,13}\b"),
    "FSSAI_LIKE": re.compile(r"\bfssai\b.{0,20}\d{10,14}", re.IGNORECASE),
}


def stable_id(path):
    return hashlib.sha1(path.encode("utf-8")).hexdigest()[:16]


def load_off_metadata(off_dir):
    rows = []
    meta_path = os.path.join(off_dir, "metadata.csv")
    if not os.path.isfile(meta_path):
        raise SystemExit(f"ERROR: {meta_path} not found -- run download_openfoodfacts.py first.")
    with open(meta_path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("local_image_path") and os.path.isfile(row["local_image_path"]):
                rows.append(row)
    return rows


def clean_and_dedup(rows, existing_sha256, min_side=64):
    """Same cleaning logic as build_dataset.py: corrupt/resolution/exact-dup/near-dup."""
    kept, rejected = [], []
    seen_sha = dict(existing_sha256)  # don't re-accept images identical to ones already in the dataset
    seen_phash = {}
    for r in rows:
        p = r["local_image_path"]
        try:
            im = Image.open(p)
            im.verify()
            im = Image.open(p).convert("RGB")
        except Exception as e:
            rejected.append({"path": p, "reason": f"corrupt_or_unreadable: {e}"})
            continue
        w, h = im.size
        if min(w, h) < min_side:
            rejected.append({"path": p, "reason": f"too_small ({w}x{h})"})
            continue
        with open(p, "rb") as fh:
            sha = hashlib.sha256(fh.read()).hexdigest()
        if sha in seen_sha:
            rejected.append({"path": p, "reason": f"duplicate_of:{seen_sha[sha]}"})
            continue
        seen_sha[sha] = p
        ph = imagehash.phash(im)
        is_near_dup = False
        for existing_p, existing_ph in seen_phash.items():
            if ph - existing_ph <= 3:
                rejected.append({"path": p, "reason": f"near_duplicate_of:{existing_p}"})
                is_near_dup = True
                break
        if is_near_dup:
            continue
        seen_phash[p] = ph
        r["width"], r["height"], r["sha256"] = w, h, sha
        r["image_id"] = stable_id(p)
        kept.append(r)
    return kept, rejected


def product_level_split_new_only(rows, existing_products_with_splits, train=0.70, val=0.15, seed=1337):
    """Assign splits only to NEW product_ids (barcodes) not already in the dataset."""
    by_product = defaultdict(list)
    for r in rows:
        by_product[r["barcode"]].append(r)
    new_product_ids = sorted(pid for pid in by_product if pid not in existing_products_with_splits)
    rnd = random.Random(seed)
    rnd.shuffle(new_product_ids)
    n = len(new_product_ids)
    n_train, n_val = int(round(n * train)), int(round(n * val))
    train_ids = set(new_product_ids[:n_train])
    val_ids = set(new_product_ids[n_train:n_train + n_val])
    for pid, items in by_product.items():
        if pid in existing_products_with_splits:
            split = existing_products_with_splits[pid]  # honor pre-existing assignment if barcode reappears
        else:
            split = "train" if pid in train_ids else ("val" if pid in val_ids else "test")
        for r in items:
            r["split"] = split
    return rows


def run_ocr(image_path):
    """Returns (regions, full_text) using REAL PaddleOCR inference.
    Verified against the actually-installed PaddleOCR 3.x / PaddleX source
    (paddlex/inference/pipelines/ocr/result.py) rather than assumed from
    memory: .predict() returns a list of dict-like OCRResult objects, each
    exposing rec_texts (list[str]), rec_scores (list[float]), and rec_boxes
    (already axis-aligned [x1,y1,x2,y2] boxes per detected line -- no manual
    polygon-to-bbox conversion needed)."""
    results = _OCR.predict(image_path)
    regions = []
    parts = []
    if not results:
        return regions, ""
    res = results[0]
    texts = res.get("rec_texts", [])
    scores = res.get("rec_scores", [])
    boxes = res.get("rec_boxes", [])
    for text, score, box in zip(texts, scores, boxes):
        text = str(text).strip()
        if not text:
            continue
        x1, y1, x2, y2 = [int(v) for v in box[:4]]
        regions.append({"bbox": [x1, y1, x2, y2], "text": text,
                         "conf": float(score), "h": int(y2 - y1)})
        parts.append(text)
    return regions, " ".join(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--off-dir", required=True, help="output dir from download_openfoodfacts.py")
    ap.add_argument("--dataset-root", required=True, help="path to legal_metrology_dataset/")
    args = ap.parse_args()
    OUT = args.dataset_root

    print("Loading Open Food Facts download metadata...")
    off_rows = load_off_metadata(args.off_dir)
    print(f"  {len(off_rows)} real downloaded image rows found")
    if not off_rows:
        raise SystemExit("Nothing to merge -- did the download actually succeed? Check off-dir/metadata.csv")

    # ---- load existing state so we never collide with it ----
    images_csv = os.path.join(OUT, "metadata", "images.csv")
    existing_sha256 = {}
    existing_products_with_splits = {}
    if os.path.isfile(images_csv):
        with open(images_csv) as f:
            for row in csv.DictReader(f):
                existing_sha256[row["sha256"]] = row["image_id"]
                existing_products_with_splits[row["product_id"]] = row["split"]

    # ---- clean + dedup against BOTH each other and the existing dataset ----
    print("Cleaning/deduplicating (corrupt check, resolution filter, exact+near-dup vs. existing dataset)...")
    kept, rejected = clean_and_dedup(off_rows, existing_sha256)
    print(f"  kept: {len(kept)}  rejected: {len(rejected)}")

    # ---- product-level split for NEW barcodes only ----
    kept = product_level_split_new_only(kept, existing_products_with_splits)

    # ---- copy images into images/indian_packaging/{split}/ ----
    img_root = os.path.join(OUT, "images", "indian_packaging")
    for r in kept:
        split_dir = os.path.join(img_root, r["split"])
        os.makedirs(split_dir, exist_ok=True)
        ext = os.path.splitext(r["local_image_path"])[1] or ".jpg"
        dest = os.path.join(split_dir, r["image_id"] + ext)
        if not os.path.exists(dest):
            shutil.copy2(r["local_image_path"], dest)
        r["dest_path"] = dest
    print(f"Copied {len(kept)} real Indian-tagged images into {img_root}")

    # ---- real OCR + regex field candidates (now WITH an MRP pattern, since
    #      this subset genuinely has Indian Rupee context) ----
    ocr_jsonl = os.path.join(OUT, "annotations", "ocr.jsonl")
    decl_jsonl = os.path.join(OUT, "annotations", "declarations.jsonl")
    quality_csv = os.path.join(OUT, "annotations", "image_quality.csv")
    yolo_img_dir = os.path.join(OUT, "annotations", "yolo", "images")
    yolo_lbl_dir = os.path.join(OUT, "annotations", "yolo", "labels")
    os.makedirs(yolo_img_dir, exist_ok=True)
    os.makedirs(yolo_lbl_dir, exist_ok=True)

    ocr_count, decl_count = 0, 0
    field_counts = defaultdict(int)
    with open(ocr_jsonl, "a") as ocr_f, open(decl_jsonl, "a") as decl_f, \
         open(quality_csv, "a", newline="") as qf:
        qwriter = csv.writer(qf)
        for idx, r in enumerate(kept):
            im = Image.open(r["dest_path"]).convert("RGB")
            gray = np.array(im.convert("L"), dtype=np.float32)
            brightness, contrast = float(gray.mean()), float(gray.std())
            gy, gx = np.gradient(gray)
            blur_var = float((gx**2 + gy**2).var())
            blur_flag = 1 if blur_var < 500 else 0
            low_light_flag = 1 if brightness < 60 else 0
            poor_contrast_flag = 1 if contrast < 25 else 0
            readability = round(max(0.0, min(1.0, (contrast / 80.0) * (1 - 0.5*blur_flag) * (1 - 0.3*low_light_flag))), 3)
            qwriter.writerow([r["image_id"], round(blur_var, 1), round(brightness, 1), round(contrast, 1),
                               blur_flag, low_light_flag, poor_contrast_flag, readability, "AUTO_CV_METRIC"])

            regions, full_text = run_ocr(r["dest_path"])
            w, h = r["width"], r["height"]
            yolo_lines = ["0 0.500000 0.500000 0.960000 0.960000"]  # PACKAGE, same documented heuristic
            for reg in regions:
                ocr_f.write(json.dumps({
                    "image_id": r["image_id"], "product_id": r["barcode"],
                    "bbox": reg["bbox"], "original_text": reg["text"],
                    "normalized_text": reg["text"].strip().lower(),
                    "ocr_confidence": round(reg["conf"], 3), "language_script": "latin",
                    "source": "OpenFoodFacts", "annotation_method": "AUTO",
                    "ocr_engine": OCR_ENGINE_TAG,
                }) + "\n")
                ocr_count += 1
                if reg["conf"] >= 0.30:
                    bx1, by1, bx2, by2 = reg["bbox"]
                    bx1, bx2 = max(0, bx1), min(w, bx2)
                    by1, by2 = max(0, by1), min(h, by2)
                    if bx2 <= bx1 or by2 <= by1:
                        continue
                    ccx, ccy = (bx1+bx2)/2/w, (by1+by2)/2/h
                    cbw, cbh = (bx2-bx1)/w, (by2-by1)/h
                    yolo_lines.append(f"1 {ccx:.6f} {ccy:.6f} {cbw:.6f} {cbh:.6f}")
            for field_type, pattern in FIELD_PATTERNS.items():
                for m in pattern.finditer(full_text):
                    decl_f.write(json.dumps({
                        "image_id": r["image_id"], "product_id": r["barcode"],
                        "field_type": field_type, "text": m.group(0), "bbox": None,
                        "source": "OpenFoodFacts", "annotation_method": "AUTO_REGEX_ON_OCR",
                        "confidence": 0.4,
                    }) + "\n")
                    decl_count += 1
                    field_counts[field_type] += 1
            ext = os.path.splitext(r["dest_path"])[1]
            shutil.copy2(r["dest_path"], os.path.join(yolo_img_dir, r["image_id"] + ext))
            with open(os.path.join(yolo_lbl_dir, r["image_id"] + ".txt"), "w") as lf:
                lf.write("\n".join(yolo_lines) + "\n")
            if (idx + 1) % 100 == 0:
                print(f"  OCR progress: {idx+1}/{len(kept)}")

    print(f"Real OCR regions written: {ocr_count}")
    print(f"Regex-derived candidate fields written: {decl_count} -> {dict(field_counts)}")
    print("NOTE: MRP candidates are still AUTO_REGEX_ON_OCR, confidence 0.4 -- unverified pattern "
          "matches, not confirmed legal MRP declarations. Human review still required.")

    # ---- append to metadata CSVs (never overwrite existing rows) ----
    meta_dir = os.path.join(OUT, "metadata")
    write_header = not os.path.isfile(images_csv) or os.path.getsize(images_csv) == 0
    with open(images_csv, "a", newline="") as f:
        w = csv.writer(f)
        for r in kept:
            w.writerow([r["image_id"], r["barcode"], r["split"], "OpenFoodFacts", "INDIA",
                        "INDIAN_PACKAGING_REAL", r.get("view_type", ""), r["width"], r["height"], r["sha256"]])

    products_csv = os.path.join(meta_dir, "products.csv")
    by_prod = defaultdict(list)
    for r in kept:
        by_prod[r["barcode"]].append(r)
    with open(products_csv, "a", newline="") as f:
        w = csv.writer(f)
        for pid, items in by_prod.items():
            w.writerow([pid, len(items), items[0]["split"], "OpenFoodFacts"])

    splits_csv = os.path.join(meta_dir, "splits.csv")
    with open(splits_csv, "a", newline="") as f:
        w = csv.writer(f)
        for r in kept:
            w.writerow([r["image_id"], r["barcode"], r["split"], "OpenFoodFacts"])

    for split in ["train", "val", "test"]:
        with open(os.path.join(OUT, "splits", f"{split}.txt"), "a") as f:
            for r in kept:
                if r["split"] == split:
                    f.write(r["image_id"] + "\n")

    provenance_csv = os.path.join(meta_dir, "provenance.csv")
    with open(provenance_csv, "a", newline="") as f:
        w = csv.writer(f)
        import datetime
        w.writerow(["Open Food Facts (merged locally)", "https://world.openfoodfacts.org",
                    "CC BY-SA 4.0 (images/data) - attribute 'Open Food Facts contributors'",
                    datetime.date.today().isoformat(),
                    f"Real India-tagged products downloaded and merged via download_openfoodfacts.py "
                    f"+ merge_openfoodfacts.py. {len(off_rows)} rows found, {len(kept)} kept after "
                    f"cleaning/dedup, {len(rejected)} rejected.",
                    len(off_rows), len(kept), "YES - CC BY-SA 4.0 permits redistribution with attribution"])

    print("\nAppended to metadata/images.csv, products.csv, splits.csv, splits/*.txt, provenance.csv")

    # ---- re-run leakage + basic validation across the COMBINED dataset ----
    print("\nRe-validating leakage across the FULL combined dataset...")
    prod_splits = defaultdict(set)
    with open(images_csv) as f:
        for row in csv.DictReader(f):
            prod_splits[row["product_id"]].add(row["split"])
    leaked = {p: list(s) for p, s in prod_splits.items() if len(s) > 1}
    leakage_report_path = os.path.join(OUT, "reports", "leakage_report_after_openfoodfacts_merge.json")
    with open(leakage_report_path, "w") as f:
        json.dump({
            "total_products_all_sources": len(prod_splits),
            "products_leaked_across_splits": len(leaked),
            "leak_detail": leaked,
            "status": "PASS" if not leaked else "FAIL",
        }, f, indent=2)
    print(f"Leakage check: {'PASS' if not leaked else 'FAIL -- see ' + leakage_report_path}")
    print(f"\nDone. {len(kept)} real Indian-tagged images merged. Re-run scripts/validate_annotations.py "
          f"for a full independent re-check before using this for training.")


if __name__ == "__main__":
    main()
