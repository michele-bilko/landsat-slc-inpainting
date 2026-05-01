"""
apply_custom_mask.py

Replace the corrupt patches in an existing NSPI dataset with new ones
generated from a custom 256x256 PNG mask. White pixels in the PNG are
treated as gaps (stripe regions); black pixels as valid data.

The clean and reference patches are unchanged; only corrupt/ is rewritten.
After running this, run_nspi.py will automatically use the new mask because
it derives the patch mask from where corrupt is zero and clean is not.

Usage:
    python apply_custom_mask.py \
        --mask_png mask_thinned.png \
        --dataset_dir data/nspi_dataset
"""
import argparse
import os
import glob
import numpy as np
from PIL import Image


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mask_png", required=True,
                   help="Path to 256x256 PNG; white = gap, black = valid")
    p.add_argument("--dataset_dir", required=True)
    p.add_argument("--threshold", type=int, default=128,
                   help="PNG grayscale threshold; > threshold counts as gap")
    args = p.parse_args()

    img = Image.open(args.mask_png).convert("L")
    arr = np.array(img)
    if arr.shape != (256, 256):
        raise SystemExit(f"Mask shape {arr.shape}, expected (256, 256)")
    mask = arr > args.threshold       # True = gap
    print(f"Loaded mask: {mask.sum()} gap pixels ({mask.mean():.1%} coverage)")

    np.save(os.path.join(args.dataset_dir, "custom_mask.npy"), mask)

    clean_dir = os.path.join(args.dataset_dir, "clean")
    corrupt_dir = os.path.join(args.dataset_dir, "corrupt")
    paths = sorted(glob.glob(os.path.join(clean_dir, "patch_*.npy")))
    print(f"Rewriting {len(paths)} corrupt patches with new mask...")
    for path in paths:
        name = os.path.basename(path)
        clean = np.load(path)         # (C, H, W)
        corrupt = clean.copy()
        corrupt[:, mask] = 0.0
        np.save(os.path.join(corrupt_dir, name), corrupt)
    print("Done. Now re-run run_nspi.py to fill with the new mask.")


if __name__ == "__main__":
    main()
