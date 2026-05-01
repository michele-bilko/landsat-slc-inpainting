import numpy as np
import os
from PIL import Image
from skimage.metrics import structural_similarity as ssim_fn

# Paths relative to this script — works for anyone who clones the repo
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EVAL_DIR   = os.path.join(SCRIPT_DIR, "patches")
CLEAN_DIR  = os.path.join(EVAL_DIR, "clean")
NR_DIR     = os.path.join(EVAL_DIR, "nr_output")
LAMA_DIR   = os.path.join(EVAL_DIR, "lama_output")
NSPI_DIR   = os.path.join(EVAL_DIR, "nspi_output")
MASK_PATH  = os.path.join(EVAL_DIR, "mask.png")

N_PATCHES = 5

# load patch handles both .npy and .png :3
def load_patch(folder, index):
    """
    Tries to load patch_{index:04d}.npy first, then .png.
    .npy: expected shape (7, 256, 256) float32
    .png: expected shape (256, 256) or (256, 256, 3) — converted to (1, 256, 256) or (3, 256, 256)
    Returns float32 array.
    """
    npy_path = os.path.join(folder, f"patch_{index:04d}.npy")
    png_path = os.path.join(folder, f"patch_{index:04d}.png")

    if os.path.exists(npy_path):
        return np.load(npy_path).astype(np.float32)
    elif os.path.exists(png_path):
        img = np.array(Image.open(png_path)).astype(np.float32)
        if img.ndim == 2:
            return img[np.newaxis, ...]          # (1, 256, 256)
        else:
            return img.transpose(2, 0, 1)        # (C, 256, 256)
    else:
        raise FileNotFoundError(
            f"No patch_{index:04d}.npy or .png found in {folder}"
        )

def load_mask(mask_path):
    """
    Load a (256, 256) PNG mask.
    White (>128) = stripe region we evaluate on.
    Returns boolean array (256, 256).
    """
    mask = np.array(Image.open(mask_path).convert("L"))
    return mask > 128

def normalize(patch):
    """
    Normalize each channel independently to [0, 1] using valid (non-zero) pixels.
    Works for any number of channels.
    """
    out = np.zeros_like(patch, dtype=np.float32)
    for c in range(patch.shape[0]):
        ch = patch[c]
        valid = ch[ch > 0]
        if valid.size == 0:
            continue
        lo, hi = valid.min(), valid.max()
        if hi - lo < 1e-6:
            continue
        out[c] = np.clip((ch - lo) / (hi - lo), 0, 1)
    return out

# 4 metric functions
def compute_psnr(clean, output, mask):
    """
    PSNR on normalized [0,1] stripe pixels only.
    Higher is better.
    """
    clean_masked  = clean[:, mask]
    output_masked = output[:, mask]
    mse = np.mean((clean_masked - output_masked) ** 2)
    if mse < 1e-10:
        return 100.0
    return float(-10 * np.log10(mse))

def compute_ssim(clean, output, mask):
    """
    SSIM averaged across channels, evaluated on stripe pixels only.
    Higher is better, max = 1.
    """
    scores = []
    for c in range(clean.shape[0]):
        score_map = ssim_fn(clean[c], output[c], data_range=1.0, full=True)[1]
        scores.append(score_map[mask].mean())
    return float(np.mean(scores))

def compute_cc(clean, output, mask):
    """
    Correlation Coefficient on stripe pixels only.
    Higher is better, max = 1.
    """
    c_vals = clean[:, mask].flatten()
    o_vals = output[:, mask].flatten()
    if c_vals.std() < 1e-6 or o_vals.std() < 1e-6:
        return 0.0
    return float(np.corrcoef(c_vals, o_vals)[0, 1])

def compute_rmse(clean, output, mask):
    """
    RMSE on normalized [0,1] stripe pixels only.
    Lower is better.
    """
    clean_masked  = clean[:, mask]
    output_masked = output[:, mask]
    return float(np.sqrt(np.mean((clean_masked - output_masked) ** 2)))

# evaluate one model across N patches
def evaluate_model(model_dir, clean_dir, mask, n_patches):
    psnrs, ssims, ccs, rmses = [], [], [], []

    for i in range(n_patches):
        try:
            clean  = normalize(load_patch(clean_dir, i))
            output = normalize(load_patch(model_dir, i))
        except FileNotFoundError as e:
            print(f"  SKIPPING patch {i}: {e}")
            continue

        # make sure channel counts match
        if clean.shape[0] != output.shape[0]:
            print(f"  WARNING patch {i}: channel mismatch "
                  f"clean={clean.shape[0]} output={output.shape[0]}, skipping.")
            continue

        if mask.sum() == 0:
            print(f"  WARNING: mask has no stripe pixels, skipping patch {i}.")
            continue

        psnr = compute_psnr(clean, output, mask)
        ssim = compute_ssim(clean, output, mask)
        cc   = compute_cc(clean, output, mask)
        rmse = compute_rmse(clean, output, mask)

        psnrs.append(psnr)
        ssims.append(ssim)
        ccs.append(cc)
        rmses.append(rmse)

        print(f"  patch {i}: PSNR={psnr:.3f}  SSIM={ssim:.3f}  "
              f"CC={cc:.3f}  RMSE={rmse:.4f}  stripe_px={mask.sum()}")

    return {
        "PSNR": float(np.mean(psnrs)),
        "SSIM": float(np.mean(ssims)),
        "CC":   float(np.mean(ccs)),
        "RMSE": float(np.mean(rmses)),
    }

print("=" * 52)
print(f"Evaluating on {N_PATCHES} patches (stripe regions only)")
print("=" * 52)

mask = load_mask(MASK_PATH)
print(f"Mask loaded: {mask.sum()} stripe pixels of {mask.size} total "
      f"({mask.mean():.1%} coverage)\n")

print("\nNo Reference:")
nr_results = evaluate_model(NR_DIR, CLEAN_DIR, mask, N_PATCHES)

print("\nLaMa:")
lama_results = evaluate_model(LAMA_DIR, CLEAN_DIR, mask, N_PATCHES)

print("\nNSPI:")
nspi_results = evaluate_model(NSPI_DIR, CLEAN_DIR, mask, N_PATCHES)

print("\n" + "=" * 52)
print(f"{'Model':<15} {'PSNR':>8} {'SSIM':>8} {'CC':>8} {'RMSE':>8}")
print("-" * 52)
print(f"{'No Reference':<15} {nr_results['PSNR']:>8.3f} {nr_results['SSIM']:>8.3f} "
      f"{nr_results['CC']:>8.3f} {nr_results['RMSE']:>8.4f}")
print(f"{'LaMa':<15} {lama_results['PSNR']:>8.3f} {lama_results['SSIM']:>8.3f} "
      f"{lama_results['CC']:>8.3f} {lama_results['RMSE']:>8.4f}")
print(f"{'NSPI':<15} {nspi_results['PSNR']:>8.3f} {nspi_results['SSIM']:>8.3f} "
      f"{nspi_results['CC']:>8.3f} {nspi_results['RMSE']:>8.4f}")
print("=" * 52)