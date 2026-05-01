"""
run_nspi.py

Run NSPI on every (clean, corrupt, reference) patch triple produced by
build_nspi_triples.py, save the filled patches, and compute evaluation metrics.

Metrics:
    - PSNR (per-band averaged)
    - SSIM (per-band averaged; needs scikit-image)
    - SAM  (Spectral Angle Mapper)
    - LPIPS (optional; needs torch + lpips. uses bands 4,3,2 = RGB)

Computed both on (a) gap pixels only (the inpainting metric) and (b) the full
patch (for comparability with DL outputs that may slightly modify non-gap
regions).

Usage:
    python run_nspi.py \
        --dataset_dir /projectnb/cs585/projects/landsat/nspi_dataset \
        --output_dir  /projectnb/cs585/projects/landsat/nspi_output \
        --window_radius 10 \
        --n_similar 20 \
        --n_workers 8 \
        --max_patches 0          # 0 = process all
"""
import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nspi import nspi_fill



def psnr_band(pred, truth, mask=None, data_range=None):
    """PSNR averaged across channels, optionally restricted to a 2D mask."""
    if mask is not None:
        diff = pred[:, mask] - truth[:, mask]
    else:
        diff = pred - truth
    if data_range is None:
        data_range = float(truth.max() - truth.min())
    if data_range == 0:
        return float("nan")
    mse = np.mean(diff ** 2, axis=tuple(range(1, diff.ndim)))  # per-channel MSE
    psnr_per_c = 20 * np.log10(data_range / (np.sqrt(mse) + 1e-12))
    return float(psnr_per_c.mean())


def sam_score(pred, truth, mask=None):
    """Spectral Angle Mapper in radians (lower = better). Mean over pixels."""
    if mask is not None:
        p = pred[:, mask]            # (C, N)
        t = truth[:, mask]
    else:
        p = pred.reshape(pred.shape[0], -1)
        t = truth.reshape(truth.shape[0], -1)
    dot = np.sum(p * t, axis=0)
    np_ = np.sqrt(np.sum(p ** 2, axis=0))
    nt_ = np.sqrt(np.sum(t ** 2, axis=0))
    cos = dot / (np_ * nt_ + 1e-12)
    cos = np.clip(cos, -1.0, 1.0)
    return float(np.mean(np.arccos(cos)))


def ssim_band(pred, truth, data_range):
    """Average SSIM across channels."""
    try:
        from skimage.metrics import structural_similarity as ssim
    except ImportError:
        return float("nan")
    vals = []
    for c in range(pred.shape[0]):
        vals.append(ssim(truth[c], pred[c], data_range=data_range))
    return float(np.mean(vals))


# Lazy LPIPS (only initialise once per worker, only if requested)
_LPIPS_MODEL = None
def lpips_rgb(pred, truth, rgb_band_idx=(2, 1, 0)):
    """LPIPS on a 3-band RGB rendering. Bands are indices into the C axis.
    For Landsat 7-channel B2..B8: RGB = (B4, B3, B2) -> indices (2, 1, 0).
    """
    global _LPIPS_MODEL
    try:
        import torch
        import lpips as lpips_lib
    except ImportError:
        return float("nan")
    if _LPIPS_MODEL is None:
        _LPIPS_MODEL = lpips_lib.LPIPS(net="alex", verbose=False)
    # Build RGB tensors in [-1, 1] using each image's own min/max
    def to_rgb(img):
        rgb = img[list(rgb_band_idx)].astype(np.float32)
        lo, hi = rgb.min(), rgb.max()
        rgb = (rgb - lo) / max(hi - lo, 1e-9)
        rgb = rgb * 2 - 1
        return torch.from_numpy(rgb).unsqueeze(0)  # (1, 3, H, W)
    with torch.no_grad():
        d = _LPIPS_MODEL(to_rgb(pred), to_rgb(truth))
    return float(d.item())



def process_one(args_tuple):
    (patch_id, clean_path, corrupt_path, ref_path, ref_cloud_path, mask_path,
     filled_dir, window_radius, n_similar, do_lpips) = args_tuple

    clean = np.load(clean_path)            # (C, H, W) float32
    corrupt = np.load(corrupt_path)
    reference = np.load(ref_path)
    mask = np.load(mask_path)              # (H, W) bool, full-scene mask

    if ref_cloud_path and os.path.exists(ref_cloud_path):
        ref_cloud = np.load(ref_cloud_path)
    else:
        ref_cloud = None

    patch_mask = np.any(corrupt == 0, axis=0) & np.any(clean != 0, axis=0)

    t0 = time.time()
    filled = nspi_fill(corrupt, reference, patch_mask,
                       ref_cloud_mask=ref_cloud,
                       window_radius=window_radius,
                       n_similar=n_similar)
    elapsed = time.time() - t0

    np.save(os.path.join(filled_dir, f"patch_{patch_id:05d}.npy"), filled)

    # Metrics on gap pixels only
    data_range = float(clean.max() - clean.min())
    res = {
        "patch_id": patch_id,
        "elapsed_s": round(elapsed, 2),
        "stripe_frac": float(patch_mask.mean()),
        "psnr_gap": psnr_band(filled, clean, mask=patch_mask, data_range=data_range),
        "sam_gap": sam_score(filled, clean, mask=patch_mask),
        "psnr_full": psnr_band(filled, clean, data_range=data_range),
        "ssim_full": ssim_band(filled, clean, data_range=data_range),
        "sam_full": sam_score(filled, clean),
    }
    if do_lpips:
        res["lpips_full"] = lpips_rgb(filled, clean)
    return res


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_dir", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--window_radius", type=int, default=10)
    p.add_argument("--n_similar", type=int, default=20)
    p.add_argument("--n_workers", type=int, default=4)
    p.add_argument("--max_patches", type=int, default=0,
                   help="0 = all; otherwise process only the first N patches "
                        "(useful for testing)")
    p.add_argument("--lpips", action="store_true",
                   help="Compute LPIPS too (requires torch + lpips package)")
    p.add_argument("--patch_ids_file", default=None,
                   help="Optional JSON list of patch_ids to evaluate "
                        "(e.g. test set). If None, all patches are processed.")
    args = p.parse_args()

    filled_dir = os.path.join(args.output_dir, "filled")
    os.makedirs(filled_dir, exist_ok=True)

    with open(os.path.join(args.dataset_dir, "metadata.json")) as f:
        meta = json.load(f)
    all_patches = meta["patches"]

    if args.patch_ids_file:
        with open(args.patch_ids_file) as f:
            allowed = set(json.load(f))
        patches = [p for p in all_patches if p["patch_id"] in allowed]
    else:
        patches = all_patches
    if args.max_patches > 0:
        patches = patches[:args.max_patches]

    print(f"Processing {len(patches)} patches with {args.n_workers} workers")

    jobs = []
    for p in patches:
        pid = p["patch_id"]
        pair_id = p["pair_id"]
        name = f"patch_{pid:05d}.npy"
        ref_cloud_dir = os.path.join(args.dataset_dir, "ref_cloud")
        ref_cloud_path = os.path.join(ref_cloud_dir, name) \
            if os.path.isdir(ref_cloud_dir) else None
        jobs.append((
            pid,
            os.path.join(args.dataset_dir, "clean", name),
            os.path.join(args.dataset_dir, "corrupt", name),
            os.path.join(args.dataset_dir, "reference", name),
            ref_cloud_path,
            os.path.join(args.dataset_dir, f"stripe_mask_pair_{pair_id:02d}.npy"),
            filled_dir,
            args.window_radius,
            args.n_similar,
            args.lpips,
        ))

    results = []
    t_all = time.time()
    if args.n_workers <= 1:
        for j in jobs:
            r = process_one(j)
            results.append(r)
            print(f"patch {r['patch_id']}: PSNR(gap)={r['psnr_gap']:.2f} "
                  f"SAM(gap)={r['sam_gap']:.4f} ({r['elapsed_s']:.1f}s)")
    else:
        with ProcessPoolExecutor(max_workers=args.n_workers) as ex:
            futs = {ex.submit(process_one, j): j[0] for j in jobs}
            for f in as_completed(futs):
                r = f.result()
                results.append(r)
                print(f"patch {r['patch_id']}: PSNR(gap)={r['psnr_gap']:.2f} "
                      f"SAM(gap)={r['sam_gap']:.4f} ({r['elapsed_s']:.1f}s)")

    print(f"\nTotal time: {time.time() - t_all:.1f}s")

    # Aggregate
    def agg(key):
        vals = [r[key] for r in results if not np.isnan(r.get(key, np.nan))]
        return float(np.mean(vals)) if vals else float("nan")

    summary = {
        "n_patches": len(results),
        "config": {
            "window_radius": args.window_radius,
            "n_similar": args.n_similar,
        },
        "mean_psnr_gap":  agg("psnr_gap"),
        "mean_sam_gap":   agg("sam_gap"),
        "mean_psnr_full": agg("psnr_full"),
        "mean_ssim_full": agg("ssim_full"),
        "mean_sam_full":  agg("sam_full"),
    }
    if args.lpips:
        summary["mean_lpips_full"] = agg("lpips_full")

    with open(os.path.join(args.output_dir, "results.json"), "w") as f:
        json.dump({"summary": summary, "per_patch": results}, f, indent=2)

    print("\n=== Summary ===")
    for k, v in summary.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
