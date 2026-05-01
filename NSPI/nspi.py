"""
nspi.py
Neighborhood Similar Pixel Interpolator (Chen et al. 2011, Remote Sensing of
Environment) for filling gaps in satellite imagery using a temporal reference.

Algorithm:
For each gap pixel p in the target image T:
  1. Open a (2K+1)x(2K+1) search window around p.
  2. From candidate pixels q in the window where BOTH T(q) and R(q) (the
     reference image) are valid:
       a. Filter to "similar" pixels: |R(q,b) - R(p,b)| < threshold for all
          bands b (Chen's coarse class filter).
       b. Rank by spectral RMSD between R(q) and R(p); keep top M.
  3. Compute weights from combined spatial + spectral distance.
  4. Two candidate predictions:
       T_hat_1(p) = sum_i w_i * T(q_i)                          (spatial)
       T_hat_2(p) = R(p) + sum_i w_i * (T(q_i) - R(q_i))        (temporal)
  5. Combine using uncertainty-based weighting (variance of similar pixels).
"""
import numpy as np


def nspi_fill(target, reference, mask,
              ref_cloud_mask=None,
              window_radius=10,
              n_similar=20,
              n_classes=4,
              spectral_threshold_factor=2.0,
              eps=1e-6,
              verbose_every=0):
    """
    Fill gap pixels in `target` using NSPI with a temporal `reference` image.

    Parameters:
    target : np.ndarray, shape (C, H, W), float32
        Target image with gaps (gap pixels are 0).
    reference : np.ndarray, shape (C, H, W), float32
        Clean reference image of the same location at a different date.
        Pixels with value exactly 0 in any band are treated as invalid.
    mask : np.ndarray, shape (H, W), bool
        True where target has gaps to fill.
    ref_cloud_mask : np.ndarray, shape (H, W), bool, optional
        True where the reference pixel is cloud / shadow / snow and should be
        treated as invalid. Combined with the zero-valued reference check.
    window_radius : int
        Half-size of the search window. (2K+1)x(2K+1) candidates. Default 10.
    n_similar : int
        Number of spectrally similar pixels to use (Chen's M). Default 20.
    n_classes : int
        Assumed number of land cover classes; controls spectral threshold.
    spectral_threshold_factor : float
        Multiplier on (sigma_band / n_classes) for the coarse similarity filter.
    eps : float
        Stability constant.
    verbose_every : int
        Print progress every N gap pixels (0 = silent).

    Returns:
    filled : np.ndarray, shape (C, H, W), float32
    """
    target = np.asarray(target, dtype=np.float32)
    reference = np.asarray(reference, dtype=np.float32)
    C, H, W = target.shape
    assert reference.shape == target.shape, \
        f"reference shape {reference.shape} != target shape {target.shape}"
    assert mask.shape == (H, W), f"mask shape {mask.shape} != ({H}, {W})"

    filled = target.copy()

    # Reference invalidity: any-band-zero OR cloud/shadow per QA mask
    ref_invalid = np.any(reference == 0, axis=0)
    if ref_cloud_mask is not None:
        ref_invalid = ref_invalid | ref_cloud_mask

    valid_ref_mask = ~ref_invalid
    if valid_ref_mask.sum() > 0:
        sigma_b = np.std(reference[:, valid_ref_mask], axis=1)
    else:
        sigma_b = np.ones(C, dtype=np.float32)
    threshold = spectral_threshold_factor * sigma_b / n_classes

    # Precomputed spatial distance grid for the window
    K = window_radius
    dy, dx = np.meshgrid(np.arange(-K, K + 1), np.arange(-K, K + 1), indexing="ij")
    spatial_dist_full = np.sqrt(dy ** 2 + dx ** 2).astype(np.float32)
    spatial_dist_full[K, K] = 1.0  # avoid zero at center (center is gap anyway)

    gap_indices = np.argwhere(mask)
    n_gaps = len(gap_indices)

    for idx_i, (yp, xp) in enumerate(gap_indices):
        y0 = max(yp - K, 0); y1 = min(yp + K + 1, H)
        x0 = max(xp - K, 0); x1 = min(xp + K + 1, W)

        win_target = target[:, y0:y1, x0:x1]
        win_ref = reference[:, y0:y1, x0:x1]
        win_gap = mask[y0:y1, x0:x1]
        win_ref_inv = ref_invalid[y0:y1, x0:x1]

        # Slice the spatial_dist grid to match the (possibly clipped) window
        sdy0 = y0 - (yp - K); sdy1 = y1 - (yp - K)
        sdx0 = x0 - (xp - K); sdx1 = x1 - (xp - K)
        win_spatial = spatial_dist_full[sdy0:sdy1, sdx0:sdx1]

        # Candidate valid: target not in gap AND reference valid
        candidate_valid = (~win_gap) & (~win_ref_inv)
        if candidate_valid.sum() < n_similar:
            # Not enough candidates; fall back to spatial mean of nearby valid target pixels.
            # Avoid using reference value at p if it's also invalid (cloud).
            target_valid_in_win = ~win_gap & np.any(win_target != 0, axis=0)
            if target_valid_in_win.sum() > 0:
                ys2, xs2 = np.where(target_valid_in_win)
                inv_d = 1.0 / (win_spatial[target_valid_in_win] + eps)
                w = inv_d / inv_d.sum()
                vals = win_target[:, ys2, xs2]
                filled[:, yp, xp] = (w[None, :] * vals).sum(axis=1)
            else:
                # Last resort: use reference if valid, otherwise leave as zero
                if not ref_invalid[yp, xp]:
                    filled[:, yp, xp] = reference[:, yp, xp]
            continue

        ref_p = reference[:, yp, xp]  # (C,)

        # If reference at the gap pixel itself is invalid (cloud), the temporal
        # prediction T_hat_2 is unreliable; fall back to spatial-only.
        ref_p_invalid = ref_invalid[yp, xp]

        # Spectral difference of reference values between candidate and target pixel
        spec_diff = win_ref - ref_p[:, None, None]                  # (C, h, w)
        within_thr = np.all(np.abs(spec_diff) < threshold[:, None, None], axis=0)
        similar_mask = candidate_valid & within_thr

        if similar_mask.sum() < n_similar:
            similar_mask = candidate_valid  # loosen if too restrictive

        spec_rmsd = np.sqrt(np.mean(spec_diff ** 2, axis=0))         # (h, w)
        rmsd_flat = spec_rmsd[similar_mask]
        spatial_flat = win_spatial[similar_mask]
        ys, xs = np.where(similar_mask)

        if len(rmsd_flat) > n_similar:
            order = np.argpartition(rmsd_flat, n_similar)[:n_similar]
            rmsd_top = rmsd_flat[order]
            spatial_top = spatial_flat[order]
            ys_top = ys[order]; xs_top = xs[order]
        else:
            rmsd_top = rmsd_flat
            spatial_top = spatial_flat
            ys_top = ys; xs_top = xs

        combined = rmsd_top * spatial_top + eps
        inv = 1.0 / combined
        weights = inv / inv.sum()                                    # (M,)

        sel_target = win_target[:, ys_top, xs_top]                   # (C, M)
        sel_ref = win_ref[:, ys_top, xs_top]                         # (C, M)

        # Spatial prediction (T_hat_1) and temporal prediction (T_hat_2)
        T_hat_1 = (weights[None, :] * sel_target).sum(axis=1)        # (C,)

        if ref_p_invalid:
            # Reference at gap pixel is invalid -> spatial only
            filled[:, yp, xp] = T_hat_1
        else:
            delta = sel_target - sel_ref                             # (C, M)
            T_hat_2 = ref_p + (weights[None, :] * delta).sum(axis=1) # (C,)

            # Combine via uncertainty (variance of similar pixels)
            var_1 = np.var(sel_target, axis=1) + eps
            var_2 = np.var(delta, axis=1) + eps
            w1 = var_2 / (var_1 + var_2)
            w2 = var_1 / (var_1 + var_2)
            filled[:, yp, xp] = w1 * T_hat_1 + w2 * T_hat_2

        if verbose_every and (idx_i + 1) % verbose_every == 0:
            print(f"  filled {idx_i+1}/{n_gaps}")

    return filled
