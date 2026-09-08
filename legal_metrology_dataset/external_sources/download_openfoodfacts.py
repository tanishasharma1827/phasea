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


def fetch_page(country, page, page_size):
    params = {
        "action": "process",
        "tagtype_0": "countries",
        "tag_contains_0": "contains",
        "tag_0": country,
        "json": 1,
        "page_size": page_size,
        "page": page,
        "fields": "code,product_name,brands,categories,countries,image_front_url,image_url",
    }
    r = requests.get(SEARCH_URL, params=params, timeout=30,
                      headers={"User-Agent": "SIH26034-LegalMetrology-StudentProject/1.0"})
    r.raise_for_status()
    return r.json().get("products", [])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--country", default="india", help="OFF country tag, e.g. india")
    ap.add_argument("--limit", type=int, default=500, help="max products to fetch")
    ap.add_argument("--out", default="./off_download")
    ap.add_argument("--page-size", type=int, default=100)
    args = ap.parse_args()

    img_dir = os.path.join(args.out, "images")
    os.makedirs(img_dir, exist_ok=True)
    meta_path = os.path.join(args.out, "metadata.csv")

    fetched = 0
    page = 1
    with open(meta_path, "w", newline="", encoding="utf-8") as mf:
        w = csv.writer(mf)
        w.writerow(["barcode", "product_name", "brand", "categories", "countries",
                    "image_url", "local_image_path", "license", "source"])
        while fetched < args.limit:
            products = fetch_page(args.country, page, args.page_size)
            if not products:
                print("No more products returned; stopping.")
                break
            for p in products:
                if fetched >= args.limit:
                    break
                code = p.get("code", "")
                img_url = p.get("image_front_url") or p.get("image_url")
                local_path = ""
                if img_url:
                    try:
                        ext = os.path.splitext(img_url)[1].split("?")[0] or ".jpg"
                        local_path = os.path.join(img_dir, f"{code}{ext}")
                        if not os.path.exists(local_path):
                            resp = requests.get(img_url, timeout=30)
                            if resp.status_code == 200:
                                with open(local_path, "wb") as imf:
                                    imf.write(resp.content)
                            else:
                                local_path = ""
                    except Exception as e:
                        print(f"  image download failed for {code}: {e}", file=sys.stderr)
                        local_path = ""
                w.writerow([code, p.get("product_name", ""), p.get("brands", ""),
                            p.get("categories", ""), p.get("countries", ""),
                            img_url or "", local_path, "CC BY-SA 4.0", "Open Food Facts"])
                fetched += 1
                if fetched % 50 == 0:
                    print(f"  fetched {fetched}/{args.limit}")
                time.sleep(0.2)  # be polite to OFF's infra
            page += 1

    print(f"\nDone. {fetched} products written to {meta_path}, images in {img_dir}/")
    print("Next: merge this into legal_metrology_dataset/images/ and re-run build_dataset.py's "
          "cleaning + product-level split logic (see scripts/build_dataset.py) to fold these in "
          "without breaking the leakage-safe split.")


if __name__ == "__main__":
    main()
