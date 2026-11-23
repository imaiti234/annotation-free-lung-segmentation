"""
Algorithm 5b (NEW) - Removal of the trachea and main bronchi.

Reference lung masks (COVID-19-CT-Seg, LCTSC/RTOG) exclude the large airways,
but a threshold method includes them because they are air.

1. Seed: searching downwards from the most superior lung-bearing slice over the
   upper 20 % of the lung range (at least 3 slices), the first 4-connected air
   component (< -900 HU) inside the body with an area of 1-8 cm^2 whose centroid
   lies less than 25 mm from the body midline (left-right direction).
2. 3D region growing (26-connectivity) of air below theta = -950, -930, -910 HU
   in turn, from the seed, until the tree reaches `airway_min_ml`. A tree larger
   than `airway_max_ml` is a leak into the lung: the search stops and the last
   smaller tree is kept (nothing is removed if there is none or no seed).
3. The tree is closed (in-plane disc, 2 px), hole-filled slice by slice and
   opened with an in-plane disc of radius `airway_min_radius_mm` to keep the
   large-calibre airways; the component containing the seed (otherwise the
   largest) is dilated by one pixel in-plane (airway wall) and removed from
   the lung masks.

(Docstring corrected for the code release; the code is unchanged.)
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi

from .alg3_separation import body_midline
from .config import Config


def _find_seed(inorm, lung_any, body, spacing, cfg):
    sz, sy, sx = spacing
    zs = np.where(lung_any.reshape(lung_any.shape[0], -1).any(1))[0]
    if zs.size == 0:
        return None
    # RAS+: higher z = superior. Search the upper 20 % of the lung range.
    top = zs[-1]
    span = max(3, int(0.2 * (zs[-1] - zs[0] + 1)))
    px_area = sy * sx / 100.0  # cm^2 per pixel
    for z in range(top, top - span, -1):
        air = (inorm[z] < cfg.airway_seed_threshold) & body[z]
        lab, n = ndi.label(air)
        if n == 0:
            continue
        mid = body_midline(body[z])
        for i, sl in enumerate(ndi.find_objects(lab), start=1):
            comp = lab[sl] == i
            area = comp.sum() * px_area
            if not (1.0 <= area <= 8.0):
                continue
            cy, cx = ndi.center_of_mass(comp)
            cx += sl[1].start
            cy += sl[0].start
            if abs(cx - mid) * sx < 25.0:
                seed = np.zeros(inorm.shape, bool)
                seed[z][sl] = comp
                return seed
    return None


def airway_mask(inorm, lung_any, body, spacing, cfg: Config) -> np.ndarray:
    seed = _find_seed(inorm, lung_any, body, spacing, cfg)
    if seed is None:
        return np.zeros(inorm.shape, bool)
    vox_ml = float(np.prod(spacing)) / 1000.0
    struct = ndi.generate_binary_structure(3, 3)
    # Grow from the seed component. Start strict (-950 HU); if the tree is
    # implausibly small (noisy lumen) relax in 20 HU steps, stopping before
    # the grown volume leaks into lung parenchyma.
    airway = np.zeros(inorm.shape, bool)
    th = cfg.airway_threshold
    while th <= cfg.airway_seed_threshold:
        air = ((inorm < th) & body) | seed
        lab, _ = ndi.label(air, structure=struct)
        ids = np.unique(lab[seed]); ids = ids[ids > 0]
        grown = np.isin(lab, ids)
        vol = grown.sum() * vox_ml
        if vol > cfg.airway_max_ml:
            break
        airway = grown
        if vol >= cfg.airway_min_ml:
            break
        th += 20
    if not airway.any():
        return airway
    # the lumen is speckled at -950 HU (noise): close and fill it slice by slice
    r = max(1, int(round(cfg.airway_min_radius_mm / min(spacing[1:]))))
    ball = np.zeros((3, 2 * r + 1, 2 * r + 1), bool)
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
    ball[1] = yy ** 2 + xx ** 2 <= r * r
    small = np.zeros((3, 5, 5), bool)
    small[1] = (np.mgrid[-2:3, -2:3] ** 2).sum(0) <= 4
    airway = ndi.binary_closing(airway, structure=small)
    for z in np.where(airway.reshape(airway.shape[0], -1).any(1))[0]:
        airway[z] = ndi.binary_fill_holes(airway[z])
    # keep only large-calibre airways (trachea, main / proximal bronchi)
    large = ndi.binary_opening(airway, structure=ball)
    lab, n = ndi.label(large, structure=struct)
    if n == 0:
        return large
    ids = np.unique(lab[seed & large]); ids = ids[ids > 0]
    large = np.isin(lab, ids) if ids.size else lab == np.argmax(np.bincount(lab.ravel())[1:]) + 1
    # airway wall: one voxel in-plane
    wall = np.zeros((1, 3, 3), bool)
    wall[0] = True
    return ndi.binary_dilation(large, structure=wall, iterations=cfg.airway_wall_px)


def remove_airways(inorm, right, left, body, spacing, cfg: Config):
    aw = airway_mask(inorm, right | left, body, spacing, cfg)
    return right & ~aw, left & ~aw, aw
