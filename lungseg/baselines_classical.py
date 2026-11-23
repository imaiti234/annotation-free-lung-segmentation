"""Classical (non-learning) comparator methods.

These are independent re-implementations of two widely used threshold-based
lung segmentation strategies. They share NO code with the proposed pipeline
(no elbow analysis, no merging, no cavity recovery), so the comparison shows
what the proposed algorithmic steps add over standard practice.

C_otsu3d   Global Otsu threshold inside the body -> 3D connected components ->
           hole filling + closing. The textbook baseline.
C_hu2001   Re-implementation in the spirit of Hu, Hoffman & Reinhardt,
           IEEE TMI 20(6):490-498, 2001: iterative optimal threshold, 3D
           connected components, trachea / main-bronchi removal by 3D region
           growing, and morphological smoothing. Evaluated as whole lung, so
           the left/right dynamic-programming separation step is not needed.

Input: HU volume (Z, Y, X), spacing (sz, sy, sx) in mm. Output: bool mask.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi
from skimage.filters import threshold_otsu

S6 = ndi.generate_binary_structure(3, 1)
S26 = ndi.generate_binary_structure(3, 3)


def _disk(r):
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
    return yy ** 2 + xx ** 2 <= r * r


def body_mask(hu: np.ndarray) -> np.ndarray:
    """Patient body per slice: largest object above -500 HU, holes filled."""
    body = np.zeros(hu.shape, bool)
    for z in range(hu.shape[0]):
        lab, n = ndi.label(hu[z] > -500)
        if n == 0:
            continue
        sizes = np.bincount(lab.ravel())
        sizes[0] = 0
        body[z] = ndi.binary_fill_holes(lab == sizes.argmax())
    return body


def _lung_components(cand: np.ndarray, keep_frac: float = 0.1) -> np.ndarray:
    lab, n = ndi.label(cand, structure=S6)
    if n == 0:
        return cand
    sizes = np.bincount(lab.ravel())
    sizes[0] = 0
    order = np.argsort(sizes)[::-1]
    keep = [order[0]]
    if n > 1 and sizes[order[1]] >= keep_frac * sizes[order[0]]:
        keep.append(order[1])     # lungs not touching each other
    return np.isin(lab, keep)


def _fill_and_close(mask: np.ndarray, spacing, close_mm: float) -> np.ndarray:
    r = max(1, int(round(close_mm / min(spacing[1:]))))
    st = _disk(r)
    out = np.zeros_like(mask)
    for z in np.where(mask.reshape(mask.shape[0], -1).any(1))[0]:
        m = np.pad(mask[z], r + 1)
        m = ndi.binary_closing(m, structure=st)[r + 1:-(r + 1), r + 1:-(r + 1)]
        out[z] = ndi.binary_fill_holes(m)
    return out


def otsu3d(hu: np.ndarray, spacing) -> np.ndarray:
    body = body_mask(hu)
    vals = np.clip(hu[body], -1024, 400)
    th = float(threshold_otsu(vals))
    cand = body & (hu < th)
    lung = _lung_components(cand)
    return _fill_and_close(lung, spacing, close_mm=3.0)


def _iterative_threshold(hu: np.ndarray, body: np.ndarray, t0: float = -500.0) -> float:
    v = hu[body].astype(np.float64)
    t = t0
    for _ in range(100):
        lo, hi = v[v < t], v[v >= t]
        if lo.size == 0 or hi.size == 0:
            break
        t_new = 0.5 * (lo.mean() + hi.mean())
        if abs(t_new - t) < 0.5:
            t = t_new
            break
        t = t_new
    return float(t)


def _trachea(hu: np.ndarray, lung: np.ndarray, spacing) -> np.ndarray:
    """3D region growing of the trachea / main bronchi from a seed on the most
    superior lung slices (RAS: higher z = superior). The growing threshold is
    raised in steps until the grown volume explodes (leak into the lungs)."""
    zs = np.where(lung.reshape(lung.shape[0], -1).any(1))[0]
    if zs.size == 0:
        return np.zeros_like(lung)
    vox_ml = float(np.prod(spacing)) / 1000.0
    px_cm2 = spacing[1] * spacing[2] / 100.0
    seed = None
    for z in range(zs[-1], max(zs[0], zs[-1] - 30) - 1, -1):
        lab, n = ndi.label(hu[z] < -900)
        for i, sl in enumerate(ndi.find_objects(lab), start=1):
            comp = lab[sl] == i
            area = comp.sum() * px_cm2
            if not 1.0 <= area <= 6.0:
                continue
            h, w = comp.shape
            if max(h, w) > 2.5 * min(h, w):        # not round
                continue
            cx = sl[1].start + w / 2.0
            if abs(cx - hu.shape[2] / 2.0) * spacing[2] > 40.0:   # not central
                continue
            seed = (z, sl, comp)
            break
        if seed is not None:
            break
    if seed is None:
        return np.zeros_like(lung)
    z, sl, comp = seed
    # slice-by-slice growing towards the feet (Hu et al. track the airway
    # in 2D): keep the < -900 HU components overlapping the airway of the
    # previous slice; stop a branch when its area jumps (leak into lung).
    region = np.zeros(hu.shape, bool)
    region[z][sl] = comp
    prev_area = comp.sum()
    for zz in range(z - 1, -1, -1):
        prev = ndi.binary_dilation(region[zz + 1], iterations=1)
        lab, n = ndi.label(hu[zz] < -900)
        ids = np.unique(lab[prev])
        ids = ids[ids > 0]
        if ids.size == 0:
            break
        cur = np.isin(lab, ids)
        area = cur.sum()
        if area > 2.0 * prev_area + 1.0 / px_cm2 or area * px_cm2 > 8.0:
            break   # leak into the lung parenchyma
        region[zz] = cur
        prev_area = area
    return ndi.binary_dilation(region, structure=S26)


def hu2001(hu: np.ndarray, spacing) -> np.ndarray:
    body = body_mask(hu)
    th = _iterative_threshold(hu, body)
    cand = body & (hu < th)
    lung = _lung_components(cand, keep_frac=0.1)
    lung &= ~_trachea(hu, lung, spacing)
    lung = _lung_components(lung, keep_frac=0.1)
    return _fill_and_close(lung, spacing, close_mm=3.0)


METHODS = {"C_otsu3d": otsu3d, "C_hu2001": hu2001}
