"""
download_pairs.py

Reads pairs.csv (output of pair_filter.py) and downloads each listed Landsat-8
scene from the USGS M2M API. Same filtering and bundle-extraction logic as
image.py — keeps only the B2..B8 TIFs and deletes everything else.

The pair structure is preserved in pairs.csv (no separate target/ vs reference/
folders) so downstream code can join scenes to pairs by product_id.

Usage:
    export USGS_USER=...
    export USGS_APP_TOKEN=...
    python download_pairs.py \
        --pairs pairs.csv \
        --download_dir /projectnb/cs585/projects/landsat/scenes \
        --bands 2 3 4 5 6 7 8
"""
import argparse
import os
import tarfile
import time
import requests
import pandas as pd

M2M = "https://m2m.cr.usgs.gov/api/api/json/stable"


def login(user, app_token):
    r = requests.post(f"{M2M}/login-token",
                      json={"username": user, "token": app_token})
    r.raise_for_status()
    data = r.json()
    if data.get("errorCode"):
        raise RuntimeError(f"Login failed: {data['errorCode']} — {data['errorMessage']}")
    print("Logged in.")
    return data["data"]


def logout(api_key):
    requests.post(f"{M2M}/logout", headers={"X-Auth-Token": api_key})
    print("Logged out.")


def get_download_url(api_key, entity_id):
    r = requests.post(f"{M2M}/download-options",
                      headers={"X-Auth-Token": api_key},
                      json={"datasetName": "landsat_ot_c2_l1",
                            "entityIds": [entity_id]})
    r.raise_for_status()
    options = r.json().get("data", [])

    bundle = next(
        (o for o in options if o.get("available") and "Bundle" in o.get("productName", "")),
        None
    )
    if bundle is None:
        bundle = next((o for o in options if o.get("available")), None)
    if bundle is None:
        raise RuntimeError(f"No available download for entity {entity_id}")

    r2 = requests.post(f"{M2M}/download-request",
                       headers={"X-Auth-Token": api_key},
                       json={"downloads": [{"entityId": entity_id,
                                            "productId": bundle["id"]}]})
    r2.raise_for_status()
    data = r2.json().get("data", {})
    urls = data.get("availableDownloads", [])
    if not urls:
        preparing = data.get("preparingDownloads", [])
        if preparing:
            raise RuntimeError(f"Download being prepared for {entity_id}. Wait and retry.")
        raise RuntimeError(f"No download URL returned for {entity_id}")
    return urls[0]["url"]


def download_scene(url, product_id, download_dir, bands_to_keep, keep_qa=True):
    tar_path = os.path.join(download_dir, f"{product_id}.tar")
    out_dir = os.path.join(download_dir, product_id)

    if os.path.exists(out_dir):
        existing = [f for f in os.listdir(out_dir) if f.endswith(".TIF")]
        expected = len(bands_to_keep) + (1 if keep_qa else 0)
        if len(existing) >= expected:
            print(f"  Already extracted ({len(existing)} TIFs), skipping.")
            return
        else:
            print(f"  Found incomplete extraction ({len(existing)} TIFs), redownloading.")

    print(f"  Downloading...")
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        downloaded = 0
        with open(tar_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                f.write(chunk)
                downloaded += len(chunk)
                if total:
                    print(f"\r  {downloaded/1e6:.0f} / {total/1e6:.0f} MB", end="")
    print()

    print(f"  Extracting...")
    os.makedirs(out_dir, exist_ok=True)
    with tarfile.open(tar_path) as tf:
        tf.extractall(out_dir)
    os.remove(tar_path)

    print(f"  Cleaning up extra files...")
    for fname in os.listdir(out_dir):
        keep_band = any(fname.endswith(f"_B{b}.TIF") for b in bands_to_keep)
        keep_qa_file = keep_qa and fname.endswith("_QA_PIXEL.TIF")
        if not (keep_band or keep_qa_file):
            os.remove(os.path.join(out_dir, fname))

    print(f"  Done -> {out_dir}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pairs", required=True, help="Path to pairs.csv from pair_filter.py")
    p.add_argument("--download_dir", required=True,
                   help="Where to put scene folders (e.g. /projectnb/cs585/projects/landsat/scenes)")
    p.add_argument("--bands", type=int, nargs="+", default=[2, 3, 4, 5, 6, 7],
                   help="Band numbers to keep (default: 2 3 4 5 6 7)")
    p.add_argument("--no_qa", action="store_true",
                   help="Skip keeping the QA_PIXEL band (cloud mask)")
    p.add_argument("--user", default=os.environ.get("USGS_USER"))
    p.add_argument("--token", default=os.environ.get("USGS_APP_TOKEN"))
    p.add_argument("--retry_sleep", type=int, default=30,
                   help="Seconds to wait when USGS says 'preparing' before retrying")
    p.add_argument("--max_retries", type=int, default=3)
    args = p.parse_args()

    if not args.user or not args.token:
        raise SystemExit("USGS_USER and USGS_APP_TOKEN must be set (env or --user/--token).")

    pairs_df = pd.read_csv(args.pairs)
    # Deduplicate scenes (same product_id might appear in multiple pairs; not currently the case,
    # but defensive)
    scenes = pairs_df[["pair_id", "role", "entity_id", "product_id"]].drop_duplicates(
        subset=["entity_id"]
    )
    print(f"Need to download {len(scenes)} unique scenes "
          f"({pairs_df['pair_id'].nunique()} pairs).")

    os.makedirs(args.download_dir, exist_ok=True)

    api_key = login(args.user, args.token)
    try:
        for i, scene in enumerate(scenes.itertuples()):
            print(f"\n[{i+1}/{len(scenes)}] pair={scene.pair_id} role={scene.role} "
                  f"product={scene.product_id}")
            print(f"  Entity ID: {scene.entity_id}")

            for attempt in range(1, args.max_retries + 1):
                try:
                    url = get_download_url(api_key, scene.entity_id)
                    download_scene(url, scene.product_id, args.download_dir,
                                   args.bands, keep_qa=not args.no_qa)
                    break
                except RuntimeError as ex:
                    if "being prepared" in str(ex) and attempt < args.max_retries:
                        print(f"  Attempt {attempt} preparing — sleeping {args.retry_sleep}s")
                        time.sleep(args.retry_sleep)
                    else:
                        print(f"  FAILED: {ex}")
                        break
                except Exception as ex:
                    print(f"  FAILED: {ex}")
                    break
    finally:
        logout(api_key)


if __name__ == "__main__":
    main()
