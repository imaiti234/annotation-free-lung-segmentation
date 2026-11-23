"""
Algorithm 4 - Adaptive merging of lesions touching the lung boundary
(manuscript Section 4.4), applied to each lung separately.

1-2. Lesion statistics (mL, stdL) are learned from lesions lying ENTIRELY inside
     the lung: pixels enclosed by the aerated lung (hole-filled minus aerated)
     whose intensity is in the lesion range. This is the paper's stated idea
     ("suspected lesion regions entirely inside the lungs ... are determined
     first"). If a slice has too few such pixels the volume-level statistics are
     used.
3-4. Candidate regions B3 = connected regions outside the lung with lesion-like
     intensity. A component B3ck is merged when
        min dist(c, lung boundary) < 15 px          (Eq. 7, c = centroid)
        (mL - 40) < mean(B3ck) < (mL + 40)           (Eq. 8)
     repeated up to 15 times (incremental growth), followed by morphological
     closing and hole filling.

Note on the original implementation: in untitled65_up.py the lesion range was
0 < I < 500 inside the thresholded lung mask, which cannot contain pixels above
the threshold, so mL was usually undefined and the merge never triggered. The
version below implements the behaviour described in the manuscript.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from scipy import ndimage as ndi

from .config import Config
from .morph import closing, fill_holes, fill_holes_selective, label2d


def internal_lesion_pixels(inorm: np.ndarray, side: np.ndarray, cfg: Config) -> np.ndarray:
    if not side.any():
        return np.empty(0, np.float32)
    enclosed = fill_holes(side) & ~side
    m = enclosed & (inorm >= cfg.lesion_low) & (inorm <= cfg.lesion_high)
    return inorm[m]


def merge_side(
    inorm: np.ndarray,
    side: np.ndarray,
    other_side: np.ndarray,
    body_filled: np.ndarray,
    stats: Optional[Tuple[float, float]],
    cfg: Config,
    px_area_mm2: Optional[float] = None,
) -> np.ndarray:
    if not side.any():
        return side

    def _fill(m):
        if px_area_mm2 is None:
            return fill_holes(m)
        return fill_holes_selective(m, inorm, cfg.hole_max_area_mm2 / px_area_mm2, cfg.lesion_high)

    current = _fill(side)
    if stats is None:
        return closing(current, cfg.closing_radius_px) & ~other_side

    m_l, s_l = stats
    lo = max(cfg.lesion_low, m_l - cfg.candidate_std_factor * s_l)
    hi = min(cfg.lesion_high, m_l + cfg.candidate_std_factor * s_l)
    pool = body_filled & ~other_side & (inorm >= lo) & (inorm <= hi)

    for _ in range(cfg.merge_iterations):
        cand = pool & ~current
        lab, n = label2d(cand)
        if n == 0:
            break
        dist = ndi.distance_transform_edt(~current)  # px distance to current lung
        idx = np.arange(1, n + 1)
        means = ndi.mean(inorm, lab, idx)
        cents = ndi.center_of_mass(cand, lab, idx)
        added = False
        for i, mc, (cy, cx) in zip(idx, means, cents):
            yy = int(np.clip(round(cy), 0, dist.shape[0] - 1))
            xx = int(np.clip(round(cx), 0, dist.shape[1] - 1))
            near = dist[yy, xx] < cfg.merge_distance_px
            similar = (m_l - cfg.merge_mean_tolerance) < mc < (m_l + cfg.merge_mean_tolerance)
            if near and similar:
                current |= lab == i
                added = True
        if not added:
            break

    current = _fill(closing(current, cfg.closing_radius_px))
    return current & ~other_side


def merge_volume(inorm_vol, right_vol, left_vol, body_vol, cfg: Config, spacing=None):
    px_area = None if spacing is None else float(spacing[1] * spacing[2])
    nz = inorm_vol.shape[0]
    slice_stats = {}
    pooled = []
    for z in range(nz):
        for name, side in (("R", right_vol[z]), ("L", left_vol[z])):
            px = internal_lesion_pixels(inorm_vol[z], side, cfg)
            if px.size >= cfg.min_internal_lesion_px:
                slice_stats[(z, name)] = (float(px.mean()), float(px.std()))
            if px.size:
                pooled.append(px)
    vol_stats = None
    if pooled:
        allpx = np.concatenate(pooled)
        if allpx.size >= cfg.min_internal_lesion_px:
            vol_stats = (float(allpx.mean()), float(allpx.std()))

    right_out = np.zeros_like(right_vol)
    left_out = np.zeros_like(left_vol)
    for z in range(nz):
        rs = slice_stats.get((z, "R"), vol_stats)
        ls = slice_stats.get((z, "L"), vol_stats)
        right_out[z] = merge_side(inorm_vol[z], right_vol[z], left_vol[z], body_vol[z], rs, cfg, px_area)
        left_out[z] = merge_side(inorm_vol[z], left_vol[z], right_out[z], body_vol[z], ls, cfg, px_area)
    return right_out, left_out, vol_stats
