#!/usr/bin/env python3
"""
Download packaging images + metadata from Open Food Facts.

WHY THIS SCRIPT ISN'T ALREADY RUN FOR YOU:
world.openfoodfacts.org and images.openfoodfacts.org are unreachable from the
sandbox this dataset was built in (network allow-list blocks them; verified
with a direct HTTPS probe -> HTTP 403). Run this on your own laptop/Colab,
where you have normal internet access, and it will work as-is.

No API key is required for read access. Please fill out OFF's API usage form
(https://openfoodfacts.github.io/openfoodfacts-server/api/#usage-form) if you
build anything beyond a college project on top of this.

Usage:
    pip install requests
    python download_openfoodfacts.py --country india --limit 500 --out ./off_download

Output:
    off_download/images/<barcode>_<n>.jpg
    off_download/metadata.csv   (barcode, product_name, brand, categories, countries, image_url, license)

License: Open Food Facts images/data are CC BY-SA 4.0. Attribute "Open Food Facts contributors".
"""
import argparse, csv, os, time, sys
import requests

SEARCH_URL = "https://world.openfoodfacts.org/cgi/search.pl"

IMAGE_FIELDS = {
    "front": "image_front_url",
    "ingredients": "image_ingredients_url",
    "nutrition": "image_nutrition_url",
}


def fetch_page(country, page, page_size, max_retries=5):
    """Retries on transient server errors (503/502/504/429) with exponential
    backoff, honoring a Retry-After header if OFF sends one. This is a real,
    documented behavior of OFF's infrastructure (their own docs: "we added
    global rate-limits... a HTTP 503 response will be returned if these
    limits are exceeded") -- not a bug on our end, so the fix is resilience,
    not a different URL."""
    params = {
        "action": "process",
        "tagtype_0": "countries",
        "tag_contains_0": "contains",
        "tag_0": country,
        "json": 1,
        "page_size": page_size,
        "page": page,
        "fields": "code,product_name,brands,categories,countries," + ",".join(IMAGE_FIELDS.values()),
    }
    last_exc = None
    for attempt in range(1, max_retries + 1):
        try:
            r = requests.get(SEARCH_URL, params=params, timeout=30,
                              headers={"User-Agent": "SIH26034-LegalMetrology-StudentProject/1.0"})
            if r.status_code in (503, 502, 504, 429):
                retry_after = r.headers.get("Retry-After")
                wait = float(retry_after) if retry_after else min(60, 3 * (2 ** (attempt - 1)))
                print(f"  page {page}: got HTTP {r.status_code} (attempt {attempt}/{max_retries}), "
                      f"waiting {wait:.0f}s before retry...", file=sys.stderr)
                last_exc = requests.exceptions.HTTPError(
                    f"{r.status_code} error on page {page} (rate-limited/unavailable)")
                time.sleep(wait)
                continue
            r.raise_for_status()
            return r.json().get("products", [])
        except requests.exceptions.RequestException as e:
            last_exc = e
            wait = min(60, 3 * (2 ** (attempt - 1)))
            print(f"  page {page}: request error ({e}) on attempt {attempt}/{max_retries}, "
                  f"waiting {wait:.0f}s before retry...", file=sys.stderr)
            time.sleep(wait)
    print(f"  page {page}: giving up after {max_retries} attempts.", file=sys.stderr)
    if last_exc:
        raise last_exc
    return []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--country", default="india", help="OFF country tag, e.g. india")
    ap.add_argument("--limit", type=int, default=500, help="max PRODUCTS to fetch (each may yield up to 3 images: front/ingredients/nutrition)")
    ap.add_argument("--out", default="./off_download")
    ap.add_argument("--page-size", type=int, default=100)
    args = ap.parse_args()

    img_dir = os.path.join(args.out, "images")
    os.makedirs(img_dir, exist_ok=True)
    meta_path = os.path.join(args.out, "metadata.csv")

    fetched_products = 0
    images_written = 0
    page = 1
    with open(meta_path, "w", newline="", encoding="utf-8") as mf:
        w = csv.writer(mf)
        # One row PER IMAGE VIEW, not per product -- this is what lets the merge
        # script build a real product-level split with multiple real images per
        # product (front/ingredients/nutrition), like GroceryStoreDataset does.
        w.writerow(["barcode", "view_type", "product_name", "brand", "categories", "countries",
                    "image_url", "local_image_path", "license", "source"])
        while fetched_products < args.limit:
            try:
                products = fetch_page(args.country, page, args.page_size)
            except requests.exceptions.RequestException as e:
                print(f"\nStopping early: page {page} failed after retries ({e}). "
                      f"Everything fetched so far ({fetched_products} products, {images_written} images) "
                      f"is already saved and usable -- this is a partial, not a corrupt, result.")
                break
            if not products:
                print("No more products returned; stopping.")
                break
            for p in products:
                if fetched_products >= args.limit:
                    break
                code = p.get("code", "")
                if not code:
                    continue
                any_image_for_this_product = False
                for view_type, field in IMAGE_FIELDS.items():
                    img_url = p.get(field)
                    if not img_url:
                        continue
                    local_path = ""
                    try:
                        ext = os.path.splitext(img_url)[1].split("?")[0] or ".jpg"
                        local_path = os.path.join(img_dir, f"{code}_{view_type}{ext}")
                        if not os.path.exists(local_path):
                            resp = requests.get(img_url, timeout=30)
                            if resp.status_code == 200:
                                with open(local_path, "wb") as imf:
                                    imf.write(resp.content)
                            else:
                                local_path = ""
                    except Exception as e:
                        print(f"  image download failed for {code}/{view_type}: {e}", file=sys.stderr)
                        local_path = ""
                    if local_path:
                        w.writerow([code, view_type, p.get("product_name", ""), p.get("brands", ""),
                                    p.get("categories", ""), p.get("countries", ""),
                                    img_url, local_path, "CC BY-SA 4.0", "Open Food Facts"])
                        images_written += 1
                        any_image_for_this_product = True
                    time.sleep(0.15)  # be polite to OFF's infra
                if any_image_for_this_product:
                    fetched_products += 1
                    if fetched_products % 50 == 0:
                        print(f"  fetched {fetched_products}/{args.limit} products, {images_written} images so far")
            page += 1
            time.sleep(1.0)  # extra pause between pages, on top of per-image pacing, to reduce rate-limit hits

    print(f"\nDone. {fetched_products} products ({images_written} images across up to 3 views each) "
          f"written to {meta_path}, images in {img_dir}/")
    print("Next: run scripts/merge_openfoodfacts.py --off-dir " + args.out +
          " --dataset-root <path to legal_metrology_dataset> to fold these into the "
          "existing leakage-safe dataset with correct provenance tags.")


if __name__ == "__main__":
    main()
