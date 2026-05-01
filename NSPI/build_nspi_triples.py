"""
build_nspi_triples.py

Produces patch *triples* (clean, corrupt, reference)
per scene pair, suitable for NSPI evaluation (and the LaMa-with-reference variant).

For each pair (target_scene, reference_scene) listed in pairs.csv:
  - Load both scenes' bands [B2..B8] -> (7, H, W) float32
  - Verify they have matching pixel grids (Landsat-8 L1TP from same WRS Path/Row should)
  - Generate the same SLC-off stripe mask as data.py (Adiyaman et al. 2024 geometry)
  - For each non-overlapping 256x256 location:
      - Extract clean patch from target scene
      - Apply stripe mask to get corrupt patch
      - Extract reference patch from reference scene at same coords
  - Discard a triple if EITHER clean or reference has >max_nodata_fraction zero pixels


Each patch_id is unique across the whole dataset; metadata.json records
which pair / scene / coords each patch came from.

Usage:
    python build_nspi_triples.py \
        --pairs pairs.csv \
        --scenes_dir /projectnb/cs585/projects/landsat/scenes \
        --out_dir /projectnb/cs585/projects/landsat/nspi_dataset \
        --bands 2 3 4 5 6 7 8 \
        --patch_size 256 \
        --max_nodata 0.10
"""
import argparse
import os
import glob
import json
import numpy as np
import pandas as pd
import rasterio
from rasterio.enums import Resampling


# Stripe geometry from Adiyaman et al. 2024
STRIPE_ANGLE_DEG = 7.5
STRIPE_EMPTY_W = 5
STRIPE_DATA_W = 70
EDGE_SCALE = 6.0


def load_qa_cloud_mask(scene_dir, target_shape):
    """Load QA_PIXEL.TIF and decode it into a boolean cloud-and-shadow mask.

    Landsat-8 Collection 2 Level 1 QA_PIXEL bit definitions:
        bit 0: Fill            bit 1: Dilated Cloud
        bit 3: Cloud           bit 4: Cloud Shadow
        bit 5: Snow            bit 7: Water (informational)

    Returns a (H, W) bool array where True = pixel should be treated as
    invalid in the reference (cloud, dilated cloud, or cloud shadow).
    Returns None if the QA file is missing.
    """
    matches = glob.glob(os.path.join(scene_dir, "*_QA_PIXEL.TIF"))
    if not matches:
        return None
    with rasterio.open(matches[0]) as src:
        if (src.height, src.width) != target_shape:
            qa = src.read(1, out_shape=target_shape,
                          resampling=Resampling.nearest)
        else:
            qa = src.read(1)
    cloud_bits = (1 << 1) | (1 << 3) | (1 << 4)  # dilated cloud | cloud | shadow
    return (qa & cloud_bits) != 0


def load_scene(scene_dir, bands):
    """Load each band TIF and stack to (C, H, W) float32.

    Bands at higher native resolution than the multispectral bands (ie.
    B8 panchromatic at 15m vs 30m for B2..B7) are downsampled to the smallest
    shape using block-mean averaging. This loses B8's spatial-resolution
    advantage but yields a uniform grid so all bands can be stacked. Use
    --bands 8 alone if you want a B8-only run at native 15m.
    """
    band_paths = {}
    band_shapes = {}
    for b in bands:
        matches = glob.glob(os.path.join(scene_dir, f"*_B{b}.TIF"))
        if not matches:
            raise FileNotFoundError(f"Band {b} not found in {scene_dir}")
        band_paths[b] = matches[0]
        with rasterio.open(matches[0]) as src:
            band_shapes[b] = (src.height, src.width)

    target_shape = (
        min(h for h, _ in band_shapes.values()),
        min(w for _, w in band_shapes.values()),
    )

    arrays = []
    for b in bands:
        with rasterio.open(band_paths[b]) as src:
            if (src.height, src.width) == target_shape:
                arr = src.read(1).astype(np.float32)
            else:
                arr = src.read(
                    1,
                    out_shape=target_shape,
                    resampling=Resampling.average,
                ).astype(np.float32)
                print(f"  Resampled B{b} from {(src.height, src.width)} "
                      f"to {target_shape}")
            arrays.append(arr)
    return np.stack(arrays, axis=0)  # (C, H, W)


def make_stripe_mask(H, W, angle_deg=STRIPE_ANGLE_DEG,
                     empty_w_center=STRIPE_EMPTY_W,
                     data_w_center=STRIPE_DATA_W,
                     edge_scale=EDGE_SCALE):
    """Boolean mask (H, W); True = stripe gap (missing data).
    Same algorithm as original data.py.
    """
    mask = np.zeros((H, W), dtype=bool)
    angle_rad = np.deg2rad(angle_deg)
    for row in range(H):
        dist_from_center = abs(row - H / 2) / (H / 2)
        empty_w = int(empty_w_center + (edge_scale - 1) * empty_w_center * dist_from_center)
        data_w = int(data_w_center)
        period = empty_w + data_w
        col_offset = int(row * np.tan(angle_rad))
        # Vectorized inner loop
        cols = np.arange(W)
        shifted = (cols - col_offset) % period
        mask[row, :] = shifted < empty_w
    return mask


def process_pair(pair_id, target_scene_dir, ref_scene_dir, bands,
                 patch_size, max_nodata_frac, out_dir, next_patch_id,
                 max_ref_cloud_frac=0.20):
    """Process one (target, reference) pair, save triples plus ref cloud mask.

    Returns (next_patch_id, list_of_metadata_entries).
    """
    print(f"\n--- Pair {pair_id} ---")
    print(f"  target:    {os.path.basename(target_scene_dir)}")
    print(f"  reference: {os.path.basename(ref_scene_dir)}")

    target = load_scene(target_scene_dir, bands)        # (C, H, W)
    ref = load_scene(ref_scene_dir, bands)              # (C, H', W')

    if target.shape != ref.shape:
        Hc = min(target.shape[1], ref.shape[1])
        Wc = min(target.shape[2], ref.shape[2])
        print(f"  shape mismatch: target={target.shape} ref={ref.shape} -> "
              f"cropping both to (C, {Hc}, {Wc})")
        target = target[:, :Hc, :Wc]
        ref = ref[:, :Hc, :Wc]

    C, H, W = target.shape
    print(f"  scene shape: ({C}, {H}, {W})")

    # Reference cloud mask from QA_PIXEL
    ref_cloud = load_qa_cloud_mask(ref_scene_dir, (H, W))
    if ref_cloud is None:
        print(f"  WARNING: no QA_PIXEL.TIF for reference, cloud masking disabled")
        ref_cloud = np.zeros((H, W), dtype=bool)
    else:
        print(f"  reference cloud fraction: {ref_cloud.mean():.1%}")

    mask = make_stripe_mask(H, W)
    coverage = mask.sum() / (H * W)
    print(f"  stripe coverage: {coverage:.1%}")

    np.save(os.path.join(out_dir, f"stripe_mask_pair_{pair_id:02d}.npy"), mask)

    clean_dir = os.path.join(out_dir, "clean")
    corrupt_dir = os.path.join(out_dir, "corrupt")
    reference_dir = os.path.join(out_dir, "reference")
    ref_cloud_dir = os.path.join(out_dir, "ref_cloud")

    metadata_entries = []
    saved = 0
    discarded_nodata = 0
    discarded_cloud = 0

    for r in range(0, H - patch_size + 1, patch_size):
        for c in range(0, W - patch_size + 1, patch_size):
            patch_clean = target[:, r:r+patch_size, c:c+patch_size]
            patch_ref = ref[:, r:r+patch_size, c:c+patch_size]
            patch_ref_cloud = ref_cloud[r:r+patch_size, c:c+patch_size]

            nodata_clean = np.max(np.sum(patch_clean == 0, axis=(1, 2))) / (patch_size**2)
            nodata_ref = np.max(np.sum(patch_ref == 0, axis=(1, 2))) / (patch_size**2)
            if nodata_clean > max_nodata_frac or nodata_ref > max_nodata_frac:
                discarded_nodata += 1
                continue

            # Discard patches whose reference has too much cloud cover
            ref_cloud_frac = patch_ref_cloud.mean()
            if ref_cloud_frac > max_ref_cloud_frac:
                discarded_cloud += 1
                continue

            patch_mask = mask[r:r+patch_size, c:c+patch_size]
            patch_corrupt = patch_clean.copy()
            patch_corrupt[:, patch_mask] = 0.0

            stripe_frac = patch_mask.sum() / (patch_size**2)

            name = f"patch_{next_patch_id:05d}.npy"
            np.save(os.path.join(clean_dir, name), patch_clean)
            np.save(os.path.join(corrupt_dir, name), patch_corrupt)
            np.save(os.path.join(reference_dir, name), patch_ref)
            np.save(os.path.join(ref_cloud_dir, name), patch_ref_cloud)

            metadata_entries.append({
                "patch_id": next_patch_id,
                "pair_id": pair_id,
                "row": int(r),
                "col": int(c),
                "target_scene": os.path.basename(target_scene_dir),
                "reference_scene": os.path.basename(ref_scene_dir),
                "nodata_clean_frac": round(float(nodata_clean), 4),
                "nodata_ref_frac": round(float(nodata_ref), 4),
                "ref_cloud_frac": round(float(ref_cloud_frac), 4),
                "stripe_frac": round(float(stripe_frac), 4),
            })
            next_patch_id += 1
            saved += 1

    print(f"  saved {saved} triples "
          f"(discarded {discarded_nodata} nodata, {discarded_cloud} cloud)")
    return next_patch_id, metadata_entries


def _json_default(o):
    """Convert numpy scalar types to JSON-native types."""
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"Object of type {type(o).__name__} is not JSON serializable")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pairs", required=True)
    p.add_argument("--scenes_dir", required=True,
                   help="Directory containing <product_id>/ folders")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--bands", type=int, nargs="+", default=[2, 3, 4, 5, 6, 7])
    p.add_argument("--patch_size", type=int, default=256)
    p.add_argument("--max_nodata", type=float, default=0.10)
    args = p.parse_args()

    pairs_df = pd.read_csv(args.pairs)

    # Set up output dirs
    for sub in ("clean", "corrupt", "reference", "ref_cloud"):
        os.makedirs(os.path.join(args.out_dir, sub), exist_ok=True)

    next_patch_id = 0
    all_metadata = []

    pair_ids = sorted(int(x) for x in pairs_df["pair_id"].unique())
    for pair_id in pair_ids:
        pair_rows = pairs_df[pairs_df["pair_id"] == pair_id]
        try:
            target_pid = pair_rows[pair_rows["role"] == "target"]["product_id"].iloc[0]
            ref_pid = pair_rows[pair_rows["role"] == "reference"]["product_id"].iloc[0]
        except IndexError:
            print(f"WARNING: pair {pair_id} missing target or reference, skipping")
            continue

        target_dir = os.path.join(args.scenes_dir, target_pid)
        ref_dir = os.path.join(args.scenes_dir, ref_pid)
        if not os.path.isdir(target_dir):
            print(f"WARNING: missing target scene dir: {target_dir}, skipping pair {pair_id}")
            continue
        if not os.path.isdir(ref_dir):
            print(f"WARNING: missing reference scene dir: {ref_dir}, skipping pair {pair_id}")
            continue

        next_patch_id, entries = process_pair(
            pair_id, target_dir, ref_dir, args.bands,
            args.patch_size, args.max_nodata, args.out_dir, next_patch_id
        )
        all_metadata.extend(entries)

    metadata_path = os.path.join(args.out_dir, "metadata.json")
    with open(metadata_path, "w") as f:
        json.dump({
            "config": {
                "bands": args.bands,
                "patch_size": args.patch_size,
                "max_nodata_frac": args.max_nodata,
                "stripe_geometry": {
                    "angle_deg": STRIPE_ANGLE_DEG,
                    "empty_w_center": STRIPE_EMPTY_W,
                    "data_w_center": STRIPE_DATA_W,
                    "edge_scale": EDGE_SCALE,
                },
            },
            "patches": all_metadata,
        }, f, indent=2, default=_json_default)
    print(f"\nWrote {len(all_metadata)} triples and metadata to {args.out_dir}/")
    print(f"  clean/      {len(all_metadata)} patches")
    print(f"  corrupt/    {len(all_metadata)} patches")
    print(f"  reference/  {len(all_metadata)} patches")


if __name__ == "__main__":
    main()
