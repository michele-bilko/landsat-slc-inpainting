"""
Visualization of NSPI fill quality on one patch (created with assistance from Claude code)

Usage:
    ./venv/bin/python visualize_patch.py 0      # patch_00000

Renders a 1x4 row: clean | corrupt | reference | filled
in true-color RGB (bands B4, B3, B2 -> indices 2, 1, 0 in our B2..B7 stack)
using a 2-98 percentile stretch on each image.
"""
import argparse
import os
import sys
import numpy as np
import matplotlib.pyplot as plt


def compute_stretch(patch, band_idx=(2, 1, 0)):
    """Compute per-channel 2-98 percentile stretch from one image.
    Returns (lo, hi) arrays of shape (3,)."""
    rgb = patch[list(band_idx)].astype(np.float32)
    los, his = [], []
    for i in range(3):
        ch = rgb[i]
        valid = ch[ch > 0]
        if len(valid) == 0:
            los.append(0.0); his.append(1.0)
            continue
        lo, hi = np.percentile(valid, [2, 98])
        los.append(lo); his.append(hi)
    return np.array(los), np.array(his)


def to_rgb(patch, los, his, band_idx=(2, 1, 0)):
    """Render an RGB image using a fixed (lo, hi) stretch passed in."""
    rgb = patch[list(band_idx)].astype(np.float32)
    out = np.zeros_like(rgb)
    for i in range(3):
        out[i] = np.clip((rgb[i] - los[i]) / max(his[i] - los[i], 1e-9), 0, 1)
    return np.transpose(out, (1, 2, 0))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("patch_id", type=int)
    p.add_argument("--dataset_dir", default="data/nspi_dataset")
    p.add_argument("--filled_dir", default="data/nspi_output/filled")
    p.add_argument("--out", default="patch_viz.png")
    args = p.parse_args()

    name = f"patch_{args.patch_id:05d}.npy"
    clean = np.load(os.path.join(args.dataset_dir, "clean", name))
    corrupt = np.load(os.path.join(args.dataset_dir, "corrupt", name))
    reference = np.load(os.path.join(args.dataset_dir, "reference", name))
    filled = np.load(os.path.join(args.filled_dir, name))

    # Shared stretch from clean: same colors mean same DN values across panels
    los, his = compute_stretch(clean)
    # Reference is from a different date/season, so it gets its own stretch
    ref_los, ref_his = compute_stretch(reference)

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    axes[0].imshow(to_rgb(clean, los, his));        axes[0].set_title("clean (truth)")
    axes[1].imshow(to_rgb(corrupt, los, his));      axes[1].set_title("corrupt (input)")
    axes[2].imshow(to_rgb(reference, ref_los, ref_his))
    axes[2].set_title("reference (different date, own stretch)")
    axes[3].imshow(to_rgb(filled, los, his));       axes[3].set_title("NSPI filled")
    for ax in axes:
        ax.axis("off")
    plt.suptitle(f"patch_{args.patch_id:05d}")
    plt.tight_layout()
    plt.savefig(args.out, dpi=120, bbox_inches="tight")
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
