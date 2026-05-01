"""
Test NSPI on a synthetic scene to make sure the implementation is correct.

Builds a fake "scene" with checkerboard-like structure, simulates a temporal
shift to make the "reference", applies a stripe mask, runs NSPI, and reports
fill quality vs (a) ground truth and (b) the trivial reference-copy baseline.
"""
import numpy as np
import sys
sys.path.insert(0, "/home/claude")
from nspi import nspi_fill


def make_synthetic_scene(H=128, W=128, C=7, seed=0):
    rng = np.random.default_rng(seed)
    yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    base = np.zeros((C, H, W), dtype=np.float32)
    for c in range(C):
        freq_y = 1 + 0.5 * c
        freq_x = 1 + 0.3 * c
        base[c] = (
            500
            + 200 * np.sin(2 * np.pi * freq_y * yy / H)
            + 150 * np.cos(2 * np.pi * freq_x * xx / W)
            + 50 * rng.standard_normal((H, W))
        )
    # "Land cover classes" — block structure
    classes = ((yy // 32) + (xx // 32)) % 4
    for c_idx in range(4):
        base[:, classes == c_idx] += 100 * c_idx
    return base.astype(np.float32)


def make_stripe_mask(H, W, period=15, gap_w=4):
    mask = np.zeros((H, W), dtype=bool)
    for c in range(W):
        if (c % period) < gap_w:
            mask[:, c] = True
    return mask


def main():
    # Ground truth scene + reference (with simulated temporal shift)
    truth = make_synthetic_scene(H=128, W=128, C=7, seed=0)

    # Reference: same scene with brightness shift + small per-pixel noise
    rng = np.random.default_rng(42)
    reference = truth + 50 + 20 * rng.standard_normal(truth.shape).astype(np.float32)
    reference = np.clip(reference, 0.1, None).astype(np.float32)  # avoid zeros

    # Stripe mask
    mask = make_stripe_mask(128, 128, period=15, gap_w=4)
    print(f"Stripe coverage: {mask.sum() / mask.size:.1%}")

    # Build corrupted target
    corrupt = truth.copy()
    corrupt[:, mask] = 0.0

    # Run NSPI
    print("Running NSPI...")
    import time
    t0 = time.time()
    filled = nspi_fill(corrupt, reference, mask,
                       window_radius=10, n_similar=20)
    print(f"NSPI took {time.time()-t0:.1f}s for {mask.sum()} gap pixels")

    # Evaluate
    def metrics(pred, name):
        diff = pred[:, mask] - truth[:, mask]
        rmse = np.sqrt(np.mean(diff ** 2))
        mae = np.mean(np.abs(diff))
        # PSNR (assuming approximate range of truth)
        data_range = truth.max() - truth.min()
        psnr = 20 * np.log10(data_range / (rmse + 1e-10))
        print(f"  {name}: RMSE={rmse:.2f}  MAE={mae:.2f}  PSNR={psnr:.2f} dB")

    print("\nMetrics on gap pixels only:")
    # Baseline 1: just copy the reference into the gaps
    baseline_copy = corrupt.copy()
    baseline_copy[:, mask] = reference[:, mask]
    metrics(baseline_copy, "ref-copy baseline")

    # Baseline 2: the corrupt image (zeros — should be terrible)
    metrics(corrupt, "leave-zero baseline")

    # NSPI
    metrics(filled, "NSPI")

    # NSPI should beat the ref-copy baseline if the temporal shift matters
    diff_nspi = np.abs(filled[:, mask] - truth[:, mask]).mean()
    diff_copy = np.abs(baseline_copy[:, mask] - truth[:, mask]).mean()
    if diff_nspi < diff_copy:
        print(f"\n✓ NSPI beats ref-copy baseline (MAE {diff_nspi:.2f} < {diff_copy:.2f})")
    else:
        print(f"\n✗ NSPI did NOT beat ref-copy (MAE {diff_nspi:.2f} >= {diff_copy:.2f})")


if __name__ == "__main__":
    main()
