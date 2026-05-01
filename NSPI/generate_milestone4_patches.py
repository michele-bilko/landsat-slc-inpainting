"""
ingest_lawrence_patches.py

Read a folder of paired PNG patches (8-bit RGB, 256x256),
with a single shared 256x256 PNG mask (white = gap, black = valid),
and produce a complete NSPI-compatible dataset on disk.

Naming convention assumed:
    <prefix>_<DATE_TARGET>_<coords>.png    (target / current — gets stripes)
    <prefix>_<DATE_REFERENCE>_<coords>.png  (reference — clean, different date)

Both files share the same <coords> suffix; pairing is on that.

Usage:
    python ingest_lawrence_patches.py \
        --patches_dir sampled_patches \
        --mask_png mask.png \
        --target_date 20210330 \
        --reference_date 20220301 \
        --out_dir data/lawrence_dataset
"""
import argparse
import glob
import json
import os
import re
import numpy as np
from PIL import Image


def load_png_chw(path):
    """Load PNG -> (3, H, W) float32."""
    arr = np.array(Image.open(path).convert("RGB")).astype(np.float32)
    return np.transpose(arr, (2, 0, 1))  # HWC -> CHW


def load_mask(path, threshold=128):
    arr = np.array(Image.open(path).convert("L"))
    return arr > threshold


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--patches_dir", required=True)
    p.add_argument("--mask_png", required=True)
    p.add_argument("--target_date", required=True,
                   help="YYYYMMDD substring identifying target (current) images")
    p.add_argument("--reference_date", required=True,
                   help="YYYYMMDD substring identifying reference images")
    p.add_argument("--out_dir", required=True)
    args = p.parse_args()

    mask = load_mask(args.mask_png)
    if mask.shape != (256, 256):
        raise SystemExit(f"Mask shape {mask.shape}, expected (256, 256)")
    print(f"Mask: {mask.sum()} gap pixels ({mask.mean():.1%} coverage)")

    for sub in ("clean", "corrupt", "reference"):
        os.makedirs(os.path.join(args.out_dir, sub), exist_ok=True)
    np.save(os.path.join(args.out_dir, "custom_mask.npy"), mask)

    # Pair files by their coordinate suffix
    target_files = sorted(glob.glob(os.path.join(
        args.patches_dir, f"*_{args.target_date}_*.png")))
    if not target_files:
        raise SystemExit(f"No target files matched *_{args.target_date}_*.png "
                         f"in {args.patches_dir}")

    coord_pat = re.compile(r"_(y\d+_x\d+)\.png$")
    metadata = []
    saved = 0

    for tgt_path in target_files:
        base = os.path.basename(tgt_path)
        m = coord_pat.search(base)
        if not m:
            print(f"  Skipping {base}: no y_x suffix")
            continue
        coords = m.group(1)
        ref_name = base.replace(args.target_date, args.reference_date)
        ref_path = os.path.join(args.patches_dir, ref_name)
        if not os.path.exists(ref_path):
            print(f"  Skipping {base}: no matching reference {ref_name}")
            continue

        target = load_png_chw(tgt_path)         # (3, 256, 256)
        reference = load_png_chw(ref_path)
        corrupt = target.copy()
        corrupt[:, mask] = 0.0

        name = f"patch_{saved:05d}.npy"
        np.save(os.path.join(args.out_dir, "clean", name), target)
        np.save(os.path.join(args.out_dir, "corrupt", name), corrupt)
        np.save(os.path.join(args.out_dir, "reference", name), reference)

        metadata.append({
            "patch_id": saved,
            "pair_id": 0,
            "coords": coords,
            "target_file": base,
            "reference_file": ref_name,
            "stripe_frac": round(float(mask.mean()), 4),
        })
        saved += 1

    with open(os.path.join(args.out_dir, "metadata.json"), "w") as f:
        json.dump({
            "config": {"bands": [1, 2, 3], "patch_size": 256,
                       "source": "lawrence_png_patches"},
            "patches": metadata,
        }, f, indent=2)

    # The stripe mask is the same for every patch with this dataset.
    # run_nspi.py looks for stripe_mask_pair_NN.npy by pair_id; satisfy that.
    np.save(os.path.join(args.out_dir, "stripe_mask_pair_00.npy"), mask)

    print(f"\nWrote {saved} patches to {args.out_dir}/")


if __name__ == "__main__":
    main()
