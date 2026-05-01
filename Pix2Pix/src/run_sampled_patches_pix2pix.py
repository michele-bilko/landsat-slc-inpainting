#!/usr/bin/env python3
"""
Run Pix2Pix on sampled_patches_pix2pix and compute report metrics.

Every PNG in sampled_patches_pix2pix is treated as a clean target unless
clean/corrupt subdirectories or clean/corrupt filename tokens are present.
For the flat clean-only layout, mask.png is applied to generate the corrupt
Pix2Pix input. White mask pixels are the hole/missing region.

The Pix2Pix baseline is a single-channel model. RGB images are handled by
running the same model independently on R, G, and B, then stacking the three
predicted channels back into a color output image.

Metrics:
  PSNR higher is better
  SSIM higher is better
  CC higher is better
  RMSE lower is better, normalized to 0-1 by dividing uint8 RMSE by 255
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import tifffile
from PIL import Image
from skimage.metrics import structural_similarity
from tqdm import tqdm

from run_pix2pix_inference import load_inference_model, model_input_spec, predict_image


DATA_RANGE = 255.0
IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")
MODEL_PATHS = {
    "A1": "model_0512_40epoch_04k.h5py",
    "A2": "model_0512_40epoch_08k.h5py",
    "A3": "model_1024_40epoch_04k.h5py",
    "A4": "model_1024_40epoch_08k.h5py",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-key", choices=("A1", "A2", "A3", "A4"), default="A1")
    parser.add_argument("--model-path", type=Path, default=None)
    parser.add_argument("--models-dir", type=Path, default=Path("models"))
    parser.add_argument("--clean-dir", type=Path, default=None)
    parser.add_argument("--corrupt-dir", type=Path, default=None)
    parser.add_argument("--mask-path", type=Path, default=None)
    parser.add_argument("--max-grids", type=int, default=16)
    return parser.parse_args()


def first_existing_dir(root: Path, names: tuple[str, ...]) -> Path | None:
    for name in names:
        path = root / name
        if path.is_dir():
            return path
    return None


def resolve_inputs(args: argparse.Namespace) -> tuple[dict[str, Path], dict[str, Path], Path]:
    clean_dir = args.clean_dir or first_existing_dir(
        args.input_root,
        ("clean", "target", "targets", "gt", "ground_truth", "groundtruth"),
    )
    corrupt_dir = args.corrupt_dir or first_existing_dir(
        args.input_root,
        ("corrupt", "masked", "input", "inputs", "slc_off", "slc-off"),
    )
    mask_path = args.mask_path or (args.input_root.parent / "mask.png")
    if not mask_path.exists():
        fallback = args.input_root / "mask.png"
        if fallback.exists():
            mask_path = fallback
    if not mask_path.exists():
        raise FileNotFoundError(f"Could not find mask.png. Tried {mask_path}")

    if clean_dir is not None and corrupt_dir is not None:
        return map_by_stem(clean_dir), map_by_stem(corrupt_dir), mask_path

    clean_map, corrupt_map = infer_flat_pairs(args.input_root)
    if clean_map:
        return clean_map, corrupt_map, mask_path

    raise FileNotFoundError(
        "Could not find sampled patch images. Put 512x512 images directly in "
        "sampled_patches_pix2pix, use clean/ and corrupt/ subdirectories, or pass "
        "--clean-dir and --corrupt-dir."
    )


def iter_images(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


def map_by_stem(root: Path) -> dict[str, Path]:
    out: dict[str, Path] = {}
    for path in iter_images(root):
        if path.name.lower() == "mask.png":
            continue
        key = path.stem
        if key in out:
            raise ValueError(f"Duplicate image stem under {root}: {key}")
        out[key] = path
    return out


def canonical_pair_id(path: Path) -> tuple[str, str | None]:
    stem = path.stem.lower()
    clean_tokens = ("clean", "target", "targets", "gt", "groundtruth", "ground_truth")
    corrupt_tokens = ("corrupt", "masked", "mask", "input", "inputs", "slc_off", "slc-off")

    for token in clean_tokens:
        for pattern in (f"_{token}_", f"-{token}-", f"_{token}", f"-{token}", f"{token}_", f"{token}-"):
            if pattern in stem:
                return stem.replace(pattern, "_"), "clean"
    for token in corrupt_tokens:
        for pattern in (f"_{token}_", f"-{token}-", f"_{token}", f"-{token}", f"{token}_", f"{token}-"):
            if pattern in stem:
                return stem.replace(pattern, "_"), "corrupt"

    parent = path.parent.name.lower()
    if parent in clean_tokens:
        return stem, "clean"
    if parent in corrupt_tokens:
        return stem, "corrupt"
    return stem, None


def infer_flat_pairs(root: Path) -> tuple[dict[str, Path], dict[str, Path]]:
    clean_map: dict[str, Path] = {}
    corrupt_map: dict[str, Path] = {}
    unknown: list[Path] = []

    for path in iter_images(root):
        if path.name.lower() == "mask.png":
            continue
        pair_id, kind = canonical_pair_id(path)
        if kind == "clean":
            clean_map[pair_id] = path
        elif kind == "corrupt":
            corrupt_map[pair_id] = path
        else:
            unknown.append(path)

    if clean_map and corrupt_map:
        return clean_map, corrupt_map
    if unknown:
        return {p.stem: p for p in unknown}, {}
    return {}, {}


def read_image(path: Path) -> np.ndarray:
    if path.suffix.lower() in (".tif", ".tiff"):
        arr = tifffile.imread(path)
    else:
        arr = np.asarray(Image.open(path).convert("RGB"))
    arr = normalize_to_uint8(arr)
    if arr.ndim == 2:
        return arr
    if arr.ndim == 3:
        return arr[..., :3]
    raise ValueError(f"Unsupported image shape for {path}: {arr.shape}")


def read_mask(path: Path) -> np.ndarray:
    if path.suffix.lower() in (".tif", ".tiff"):
        arr = tifffile.imread(path)
    else:
        arr = np.asarray(Image.open(path).convert("L"))
    return normalize_to_uint8(arr)


def normalize_to_uint8(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.dtype == np.uint8:
        return arr
    arr = arr.astype(np.float32)
    finite = np.isfinite(arr)
    if not np.any(finite):
        return np.zeros(arr.shape, dtype=np.uint8)
    vals = arr[finite]
    lo = float(vals.min())
    hi = float(vals.max())
    if hi <= 1.0 and lo >= 0.0:
        arr = arr * 255.0
    elif hi > 255.0 or lo < 0.0:
        p2, p98 = np.percentile(vals, [2, 98])
        if p98 > p2:
            arr = (arr - p2) * (255.0 / (p98 - p2))
    arr = np.clip(arr, 0, 255)
    arr[~finite] = 0
    return arr.astype(np.uint8)


def write_png(path: Path, arr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8)).save(path)


def resize_mask(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    if mask.shape == shape:
        return mask
    img = Image.fromarray(mask.astype(np.uint8) * 255)
    img = img.resize((shape[1], shape[0]), resample=Image.Resampling.NEAREST)
    return np.asarray(img) > 127


def mask_for_image(mask: np.ndarray, image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return mask
    return np.repeat(mask[:, :, None], image.shape[2], axis=2)


def masked_values(clean: np.ndarray, pred: np.ndarray, mask: np.ndarray | None) -> tuple[np.ndarray, np.ndarray]:
    if mask is None:
        return clean.reshape(-1).astype(np.float32), pred.reshape(-1).astype(np.float32)
    active = mask_for_image(mask, clean).reshape(-1)
    return clean.reshape(-1).astype(np.float32)[active], pred.reshape(-1).astype(np.float32)[active]


def psnr(clean: np.ndarray, pred: np.ndarray, mask: np.ndarray | None) -> float:
    x, y = masked_values(clean, pred, mask)
    if x.size == 0:
        return math.nan
    mse = float(np.mean((x - y) ** 2))
    if mse <= 1e-12:
        return math.inf
    return float(20.0 * math.log10(DATA_RANGE / math.sqrt(mse)))


def ssim_metric(clean: np.ndarray, pred: np.ndarray, mask: np.ndarray | None) -> float:
    kwargs: dict[str, Any] = {}
    if clean.ndim == 3:
        kwargs["channel_axis"] = -1
    score, ssim_map = structural_similarity(clean, pred, data_range=DATA_RANGE, full=True, **kwargs)
    if mask is None:
        return float(score)
    if not np.any(mask):
        return math.nan
    active = mask_for_image(mask, ssim_map) if ssim_map.ndim == 3 else mask
    return float(np.mean(ssim_map[active]))


def cc(clean: np.ndarray, pred: np.ndarray, mask: np.ndarray | None) -> float:
    x, y = masked_values(clean, pred, mask)
    if x.size == 0:
        return math.nan
    x = x.astype(np.float64)
    y = y.astype(np.float64)
    x = x - x.mean()
    y = y - y.mean()
    denom = math.sqrt(float(np.mean(x * x) * np.mean(y * y)))
    if denom <= 1e-12:
        return math.nan
    return float(np.mean(x * y) / denom)


def rmse_normalized(clean: np.ndarray, pred: np.ndarray, mask: np.ndarray | None) -> float:
    x, y = masked_values(clean, pred, mask)
    if x.size == 0:
        return math.nan
    return float(np.sqrt(np.mean(((x - y) / DATA_RANGE) ** 2)))


def save_grid(path: Path, clean: np.ndarray, corrupt: np.ndarray, pred: np.ndarray, mask: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 4, figsize=(14, 4), constrained_layout=True)
    items = (
        (clean, "Clean"),
        (corrupt, "Corrupt"),
        (pred, "Pix2Pix"),
        (mask.astype(np.uint8) * 255, "Mask"),
    )
    for ax, (arr, title) in zip(axes, items):
        if arr.ndim == 2:
            ax.imshow(arr, cmap="gray", vmin=0, vmax=255)
        else:
            ax.imshow(arr, vmin=0, vmax=255)
        ax.set_title(title)
        ax.axis("off")
    fig.savefig(path, dpi=160)
    plt.close(fig)


def mean_finite(values: list[float]) -> float:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    return float(arr.mean()) if arr.size else math.nan


def metric_row(prefix: str, clean: np.ndarray, pred: np.ndarray, mask: np.ndarray | None) -> dict[str, float]:
    return {
        f"{prefix}_psnr": psnr(clean, pred, mask),
        f"{prefix}_ssim": ssim_metric(clean, pred, mask),
        f"{prefix}_cc": cc(clean, pred, mask),
        f"{prefix}_rmse": rmse_normalized(clean, pred, mask),
    }


def predict_color_image(
    model: Any,
    image: np.ndarray,
    model_h: int | None,
    model_w: int | None,
    expects_channel: bool,
) -> np.ndarray:
    if image.ndim == 2:
        return predict_image(model, image, model_h, model_w, expects_channel)

    channels = []
    for c in range(image.shape[2]):
        pred_c = predict_image(model, image[..., c], model_h, model_w, expects_channel)
        channels.append(pred_c)
    return np.stack(channels, axis=2)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pred_dir = args.output_dir / f"predictions_{args.model_key}"
    corrupt_out_dir = args.output_dir / f"corrupt_inputs_{args.model_key}"
    grids_dir = args.output_dir / f"visual_grids_{args.model_key}"
    metrics_csv = args.output_dir / f"metrics_{args.model_key}.csv"
    summary_csv = args.output_dir / f"metrics_summary_{args.model_key}.csv"
    manifest_csv = args.output_dir / f"inference_manifest_{args.model_key}.csv"

    clean_map, corrupt_map, mask_path = resolve_inputs(args)
    generated_corrupt = not corrupt_map
    common = sorted(clean_map) if generated_corrupt else sorted(set(clean_map) & set(corrupt_map))
    if not common:
        raise RuntimeError("No sampled patch images found to evaluate")

    model_path = args.model_path or (args.models_dir / MODEL_PATHS[args.model_key])
    model = load_inference_model(model_path)
    model_h, model_w, expects_channel = model_input_spec(model)
    mask_base = read_mask(mask_path) > 127

    rows: list[dict[str, Any]] = []
    with manifest_csv.open("w", newline="") as mf:
        manifest_writer = csv.DictWriter(
            mf,
            fieldnames=["image_id", "clean_path", "corrupt_path", "prediction_path", "height", "width"],
        )
        manifest_writer.writeheader()

        for idx, image_id in enumerate(tqdm(common, desc=f"Pix2Pix {args.model_key}")):
            clean = read_image(clean_map[image_id])
            mask = resize_mask(mask_base, clean.shape[:2])
            if generated_corrupt:
                corrupt = clean.copy()
                if corrupt.ndim == 2:
                    corrupt[mask] = 0
                else:
                    corrupt[mask, :] = 0
                corrupt_path = corrupt_out_dir / f"{image_id}_corrupt.png"
                write_png(corrupt_path, corrupt)
            else:
                corrupt = read_image(corrupt_map[image_id])
                corrupt_path = corrupt_map[image_id]
                if clean.shape != corrupt.shape:
                    raise ValueError(f"Shape mismatch for {image_id}: clean {clean.shape}, corrupt {corrupt.shape}")

            pred = predict_color_image(model, corrupt, model_h, model_w, expects_channel)
            pred = np.clip(pred, 0, 255).astype(np.uint8)
            pred_path = pred_dir / f"{image_id}.png"
            write_png(pred_path, pred)

            row: dict[str, Any] = {
                "image_id": image_id,
                "height": clean.shape[0],
                "width": clean.shape[1],
                "channels": 1 if clean.ndim == 2 else clean.shape[2],
                "mask_fraction": float(mask.mean()),
            }
            row.update(metric_row("pix2pix_full", clean, pred, None))
            row.update(metric_row("pix2pix_hole", clean, pred, mask))
            row.update(metric_row("corrupt_full", clean, corrupt, None))
            row.update(metric_row("corrupt_hole", clean, corrupt, mask))
            rows.append(row)

            manifest_writer.writerow(
                {
                    "image_id": image_id,
                    "clean_path": clean_map[image_id],
                    "corrupt_path": corrupt_path,
                    "prediction_path": pred_path,
                    "height": clean.shape[0],
                    "width": clean.shape[1],
                }
            )

            if idx < args.max_grids:
                save_grid(grids_dir / f"{image_id}_grid.png", clean, corrupt, pred, mask)

    fieldnames = ["image_id"] + sorted({k for row in rows for k in row.keys() if k != "image_id"})
    with metrics_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    with summary_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["metric", "mean"])
        writer.writeheader()
        for key in fieldnames:
            if key == "image_id":
                continue
            vals = [row.get(key, math.nan) for row in rows]
            numeric_vals = [float(v) for v in vals if isinstance(v, (float, int, np.floating))]
            writer.writerow({"metric": key, "mean": mean_finite(numeric_vals)})

    print(f"Wrote corrupt inputs to {corrupt_out_dir}")
    print(f"Wrote predictions to {pred_dir}")
    print(f"Wrote metrics to {metrics_csv}")
    print(f"Wrote summary to {summary_csv}")
    print(f"Wrote grids to {grids_dir}")


if __name__ == "__main__":
    main()
