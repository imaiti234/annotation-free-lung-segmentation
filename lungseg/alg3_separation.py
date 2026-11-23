"""
Algorithm 3 - Left / right lung separation (manuscript Section 4.3).

1-3. From Lung_refined3 take L1, L2 (two largest). If R = |L2|/|L1| >= 0.3 they
     are already separate.
4.   Otherwise lower the threshold in steps of 5 until two comparable regions
     (L3, L4) appear; Lung_difference = Lung_refined3 minus (L3 u L4). Each
     component of the difference is dilated (5x) and discarded if it would
     reconnect L3 and L4 (a bridge); all others are restored (Eq. 4).
5.   Left / right assignment by centroid position.

Robustness additions (documented in the paper as implementation details):
 * a candidate split is accepted only when L3 and L4 lie on opposite sides of
   the body midline (prevents a single lung being split at apex/base slices);
 * if no split is found, components crossing the midline are cut along it.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np
from scipy import ndimage as ndi

from .alg1_coarse import segment_at_threshold
from .config import Config
from .morph import components_sorted, label2d, remove_small


def body_midline(body_filled: np.ndarray) -> float:
    cols = np.nonzero(body_filled)[1]
    return float(cols.mean()) if cols.size else body_filled.shape[1] / 2.0


def _centroid_x(mask: np.ndarray) -> float:
    return float(np.nonzero(mask)[1].mean())


def _ratio(comps) -> float:
    if len(comps) < 2:
        return 0.0
    return comps[1].sum() / max(comps[0].sum(), 1)


def _assign_sides(lungsep, midline, cfg) -> Tuple[np.ndarray, np.ndarray]:
    """RAS+ convention: larger column index = patient's right."""
    right = np.zeros_like(lungsep, bool)
    left = np.zeros_like(lungsep, bool)
    cols = np.arange(lungsep.shape[1])[None, :]
    for comp in components_sorted(lungsep):
        frac_right = (comp & (cols > midline)).sum() / comp.sum()
        crosses = 0.2 < frac_right < 0.8
        if crosses and cfg.midline_split_fallback:
            right |= comp & (cols > midline)
            left |= comp & (cols <= midline)
        elif _centroid_x(comp) > midline:
            right |= comp
        else:
            left |= comp
    return right, left


def separate_slice(inorm, refined3, th_star, body_filled, cfg: Config):
    """Return (right, left, used_threshold_lowering)."""
    if not refined3.any():
        z = np.zeros_like(refined3, bool)
        return z, z.copy(), False
    midline = body_midline(body_filled)
    comps = components_sorted(refined3)
    if _ratio(comps) >= cfg.separation_ratio:
        r, l = _assign_sides(comps[0] | comps[1], midline, cfg)
        return r, l, False

    # Only genuinely joined lungs need the threshold-lowering loop: if no
    # component crosses the body midline, the regions are already one per side
    # (e.g. apex/base slices with a single lung).
    cols = np.arange(refined3.shape[1])[None, :]
    crossing = any(0.1 < (c & (cols > midline)).sum() / c.sum() < 0.9 for c in comps[:2])
    if not crossing:
        r, l = _assign_sides(comps[0] | (comps[1] if len(comps) > 1 else comps[0]), midline, cfg)
        return r, l, False

    th2 = th_star
    temp = None
    while th2 - cfg.separation_step >= cfg.separation_min_threshold:
        th2 -= cfg.separation_step
        lung = segment_at_threshold(inorm, th2, cfg).lung & refined3
        cs = components_sorted(lung)
        if _ratio(cs) >= cfg.separation_ratio:
            same_side = (_centroid_x(cs[0]) > midline) == (_centroid_x(cs[1]) > midline)
            if not same_side:
                l3, l4 = cs[0], cs[1]
                temp = l3 | l4
                break

    if temp is None:
        r, l = _assign_sides(refined3, midline, cfg)
        return r, l, True

    diff = remove_small(refined3 & ~temp, 4)  # original code: drop components <= 3 px
    kernel = np.ones((cfg.bridge_kernel_size, cfg.bridge_kernel_size), bool)
    keep = np.zeros_like(diff)
    lab, n = label2d(diff)
    for i in range(1, n + 1):
        cc = lab == i
        dil = ndi.binary_dilation(cc, structure=kernel, iterations=cfg.bridge_dilation_iterations)
        bridges = (dil & l3).any() and (dil & l4).any()
        if not bridges:
            keep |= cc
    lungsep = temp | keep
    r, l = _assign_sides(lungsep, midline, cfg)
    return r, l, True


def separate_volume(inorm_vol, refined3_vol, thresholds, body_vol, cfg: Config):
    right = np.zeros(refined3_vol.shape, bool)
    left = np.zeros_like(right)
    lowered = np.zeros(refined3_vol.shape[0], bool)
    for z in range(refined3_vol.shape[0]):
        right[z], left[z], lowered[z] = separate_slice(
            inorm_vol[z], refined3_vol[z], thresholds[z], body_vol[z], cfg
        )
    return right, left, lowered
