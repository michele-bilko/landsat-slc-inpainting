# Multi-Spectral Inpainting for Landsat-7 SLC-Off Image Reconstruction

Final project for CS 585 (Image and Video Computing), Boston University.

**Team:** Michele Bilko, Olivia Ma, Lawrence Miao, Prem Rajendran

## Background

The Landsat-7 satellite has been imaging Earth since 1999. In 2003, a hardware
component called the Scan Line Corrector (SLC) failed permanently. Every image
since has stripe-shaped gaps covering ~22% of each scene. We treat recovering
the missing data as a multi-spectral image inpainting problem and compare
several reconstruction methods.

## Methods compared

- **Pix2Pix GAN** (Adıyaman et al., 2024) — single-band deep learning baseline
- **NSPI** (Chen et al., 2011) — non-deep-learning baseline using a clean
  reference image of the same location at a different date
- **LaMa (no reference)** — large-mask inpainting model, no temporal reference
- **LaMa with reference** — our proposed extension; LaMa with the reference
  image stacked as additional input channels

## Setup

```bash
git clone https://github.com/<owner>/<repo>.git
cd <repo>

python3 -m venv venv
source venv/bin/activate    # Windows: venv\Scripts\activate
pip install numpy pandas rasterio scikit-image requests pillow matplotlib torch lpips
```

For the download step you also need USGS credentials:

```bash
export USGS_USER=<your_usgs_username>
export USGS_APP_TOKEN=<your_token>
```

## NSPI baseline pipeline

NSPI (Neighborhood Similar Pixel Interpolator) fills each missing pixel by
finding spectrally similar neighbors in a clean reference image and blending
their values.

The pipeline has five stages, run as separate scripts.

## Notes on the data

We do not commit raw Landsat scenes, downloaded products, generated patches,
or NSPI output `.npy` files to git — they are too large. The `data/` folder is
gitignored. Re-running the download and patch-build scripts will reproduce the
dataset exactly given the same `pairs.csv` and seed.

The `sampled_patches/` folder is the agreed-upon evaluation set;
all four methods are evaluated on these same patches for the comparison table.

## Metric conventions

- **PSNR** — Peak Signal-to-Noise Ratio, dB, higher is better
- **SSIM** — Structural Similarity Index, 0–1, higher is better
- **CC** — Correlation Coefficient, 0–1, higher is better
- **RNSE** — Root Mean Squared Error, normalized 0–1, lower is better

All metrics are computed on the masked (gap) region only, evaluating reconstruction quality where data was missing.

## References

- Adıyaman, H., Varul, Y. E., Bakırman, T., Bayram, B. (2024). Stripe error
  correction for Landsat-7 using deep learning. *PFG – Journal of
  Photogrammetry, Remote Sensing and Geoinformation Science*, 93, 51–63.
- Chen, J., Zhu, X., Vogelmann, J. E., Gao, F., Jin, S. (2011). A simple and
  effective method for filling gaps in Landsat ETM+ SLC-off images.
  *Remote Sensing of Environment*, 115(4), 1053–1064.
- Suvorov, R., et al. (2022). Resolution-robust Large Mask Inpainting with
  Fourier Convolutions. *WACV 2022*.