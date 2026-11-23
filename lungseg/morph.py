"""Small binary-morphology helpers shared by the algorithms."""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi
from skimage.morphology import disk

STRUCT2D = ndi.generate_binary_structure(2, 2)  # 8-connectivity
STRUCT3D = ndi.generate_binary_structure(3, 1)  # 6-connectivity


def label2d(mask):
    return ndi.label(mask, structure=STRUCT2D)


def remove_small(mask: np.ndarray, min_px: int) -> np.ndarray:
    lab, n = label2d(mask)
    if n == 0:
        return mask.astype(bool)
    sizes = np.bincount(lab.ravel())
    keep = sizes >= min_px
    keep[0] = False
    return keep[lab]


def largest_k(mask: np.ndarray, k: int) -> np.ndarray:
    """Union of the k largest 8-connected components."""
    lab, n = label2d(mask)
    if n == 0:
        return np.zeros_like(mask, dtype=bool)
    sizes = np.bincount(lab.ravel())
    sizes[0] = 0
    order = np.argsort(sizes)[::-1][: min(k, n)]
    order = order[sizes[order] > 0]
    return np.isin(lab, order)


def components_sorted(mask: np.ndarray):
    """List of boolean component masks, largest first."""
    lab, n = label2d(mask)
    if n == 0:
        return []
    sizes = np.bincount(lab.ravel())
    sizes[0] = 0
    order = np.argsort(sizes)[::-1]
    return [lab == i for i in order if sizes[i] > 0]


def fill_holes(mask: np.ndarray) -> np.ndarray:
    return ndi.binary_fill_holes(mask)


def closing(mask: np.ndarray, radius_px: int) -> np.ndarray:
    if radius_px <= 0 or not mask.any():
        return mask.astype(bool)
    pad = radius_px + 1
    m = np.pad(mask, pad)
    m = ndi.binary_closing(m, structure=disk(radius_px))
    return m[pad:-pad, pad:-pad]


def boundary(mask: np.ndarray) -> np.ndarray:
    return mask & ~ndi.binary_erosion(mask, structure=STRUCT2D)


def fill_holes_selective(mask: np.ndarray, image: np.ndarray, max_area_px: float,
                         max_mean: float) -> np.ndarray:
    """Fill only holes that are small (vessels, bronchi) or lesion-like
    (mean intensity <= max_mean). Large dense holes - the liver dome or the
    heart cut obliquely - are left open; truly enclosed consolidation is
    recovered later by 3D hole filling."""
    holes = fill_holes(mask) & ~mask
    if not holes.any():
        return mask.astype(bool)
    lab, n = label2d(holes)
    area = np.bincount(lab.ravel(), minlength=n + 1).astype(np.float64)
    total = np.bincount(lab.ravel(), weights=image.ravel().astype(np.float64), minlength=n + 1)
    mean = total / np.maximum(area, 1)
    keep = (area <= max_area_px) | (mean <= max_mean)
    keep[0] = False
    return mask.astype(bool) | keep[lab]
