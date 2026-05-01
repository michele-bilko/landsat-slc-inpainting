import pandas as pd
import requests
import os
import tarfile

USGS_USER = os.environ.get("USGS_USER", "")
USGS_APP_TOKEN = os.environ.get("USGS_APP_TOKEN", "")

if not USGS_USER or not USGS_APP_TOKEN:
    raise RuntimeError("Please set USGS_USER and USGS_APP_TOKEN environment variables.")

CSV_PATH     = os.environ.get("CSV_PATH", "data/scenes.csv")
DOWNLOAD_DIR = os.environ.get("DOWNLOAD_DIR", "data/scenes")

M2M = "https://m2m.cr.usgs.gov/api/api/json/stable"

os.makedirs(DOWNLOAD_DIR, exist_ok=True)

# Option A: test with a single Entity ID (uncomment to use)
entity_ids  = ["LC80160422021060LGN00"]
product_ids = ["LC08_L1TP_016042_20210301_20210311_02_T1"]

# Option B: load all IDs from CSV (comment out Option A first)
# df = pd.read_csv(CSV_PATH)
# product_ids = df["Landsat Product Identifier L1"].astype(str).tolist()
# entity_ids  = df["Entity ID"].astype(str).tolist()
# print(f"Found {len(product_ids)} scenes in CSV")

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

def download_scene(url, product_id, download_dir):
    tar_path = os.path.join(download_dir, f"{product_id}.tar")
    out_dir = os.path.join(download_dir, product_id)

    if os.path.exists(out_dir):
        print(f"  Already extracted, skipping.")
        return

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

    # For each scene there are multiple files, but we only need a subset. 
    # specifically, B2 through B8 extension would be _B2.TIF - _B8.TIF. 
    # For a total of 7 TIF files per scene
    # Update!!!! Keep only bands 2-8, delete everything else
    print(f"  Cleaning up extra files...")
    for fname in os.listdir(out_dir):
        keep = any(fname.endswith(f"_B{b}.TIF") for b in range(2, 9))
        if not keep:
            os.remove(os.path.join(out_dir, fname))
            print(f"    Deleted: {fname}")
            
    print(f"  Done → {out_dir}")

# Main execution
api_key = login(USGS_USER, USGS_APP_TOKEN)

try:
    for i, (product_id, entity_id) in enumerate(zip(product_ids, entity_ids)):
        print(f"\n[{i+1}/{len(product_ids)}] {product_id}")
        print(f"  Entity ID: {entity_id}")
        try:
            url = get_download_url(api_key, entity_id)
            download_scene(url, product_id, DOWNLOAD_DIR)
        except Exception as ex:
            print(f"  FAILED: {ex}")
finally:
    logout(api_key)