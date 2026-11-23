"""
Algorithm 4c (NEW) - Rib-cage radial envelope for dense consolidation.

Problem
-------
Dense consolidation (about -100 ... +60 HU) has the same density as the chest
wall, so no threshold (Algorithms 1-2) and no intensity window (Algorithm 4b)
can separate it. It is, however, *geometrically* constrained: the lung fills the
thoracic cavity up to the inner surface of the ribs. A consolidated region shows
up as a thick gap between the aerated lung and the ribs.

Method (per lung, per slice; polar coordinates around the lung centroid)
------------------------------------------------------------------------
For every ray angle theta:
  r_L(theta)  outermost lung pixel on the ray
  r_B(theta)  first bone pixel beyond r_L (ribs, vertebra)
  r_R(theta)  inner rib-cage radius = min of r_B over a +/- `window_deg`
              neighbourhood (bridges the intercostal spaces, ignores the
              scapula, which lies outside the ribs)
The segment r_L ... r_R - margin is added when
  * the gap is at least `min_gap_mm` thick (thin gaps are normal pleura),
  * the gap contains (almost) no fat  (-190 ... -60 HU)  -> excludes extrapleural,
    pericardial and paravertebral fat, and the heart, which is wrapped in fat,
  * the ray does not point into the mediastinum (medial/antero-medial sector,
    defined from the vertebra position, so it works for any patient orientation),
  * the gap stays inside the body and does not touch the other lung.
Added pixels must also lie in a plausible parenchyma range (< +150 HU, not bone),
and every added 3D blob must span >= 2 slices.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi
from scipy.ndimage import minimum_filter1d

from .config import Config
from .morph import closing, fill_holes, label2d, remove_small


def _spine_centroid(bone, midline, sx):
    """Largest bone component near the midline = vertebra (sternum is smaller)."""
    lab, n = label2d(bone)
    if n == 0:
        return None
    best, best_size = None, 0
    for i, sl in enumerate(ndi.find_objects(lab), start=1):
        comp = lab[sl] == i
        size = int(comp.sum())
        cy, cx = ndi.center_of_mass(comp)
        cx += sl[1].start
        cy += sl[0].start
        if abs(cx - midline) * sx < 30.0 and size > best_size:
            best, best_size = (cy, cx), size
    return best


def _rays(cy, cx, shape, n_angles, r_max):
    th = np.deg2rad(np.arange(n_angles) * 360.0 / n_angles)
    r = np.arange(r_max, dtype=np.float32)
    yy = cy + np.sin(th)[:, None] * r[None, :]
    xx = cx + np.cos(th)[:, None] * r[None, :]
    inside = (yy >= 0) & (yy < shape[0] - 0.5) & (xx >= 0) & (xx < shape[1] - 0.5)
    yi = np.clip(np.rint(yy), 0, shape[0] - 1).astype(np.int32)
    xi = np.clip(np.rint(xx), 0, shape[1] - 1).astype(np.int32)
    return th, yi, xi, inside


def _envelope_side(inorm, side, other, bone, body, midline, spine, spacing_yx, cfg: Config):
    sy, sx = spacing_yx
    px = float(min(sy, sx))
    if side.sum() < 50:
        return np.zeros_like(side)
    cy, cx = ndi.center_of_mass(side)
    r_max = int(np.hypot(*side.shape))
    n_ang = int(round(360.0 / cfg.costal_angle_step_deg))
    th, yi, xi, inside = _rays(cy, cx, side.shape, n_ang, r_max)

    lung_r = side[yi, xi] & inside
    bone_r = bone[yi, xi] & inside
    body_r = body[yi, xi] & inside
    other_r = other[yi, xi] & inside
    hu_r = inorm[yi, xi]

    idx = np.arange(r_max)[None, :]
    has_lung = lung_r.any(1)
    r_l = np.where(has_lung, r_max - 1 - np.argmax(lung_r[:, ::-1], axis=1), -1)

    beyond = bone_r & (idx > r_l[:, None])
    has_bone = beyond.any(1)
    r_b = np.where(has_bone, np.argmax(beyond, axis=1), r_max).astype(np.float32)
    max_gap = cfg.costal_max_gap_mm / px
    r_b[(r_b - r_l) > max_gap] = r_max
    win = max(1, int(round(cfg.costal_window_deg / cfg.costal_angle_step_deg)))
    r_rib = minimum_filter1d(r_b, size=2 * win + 1, mode="wrap")

    # mediastinal sector, from the vertebra (orientation independent)
    medial_x = np.sign(midline - cx) if abs(midline - cx) > 1 else 0.0
    d_y, d_x = np.sin(th), np.cos(th)
    medial = d_x * medial_x
    if spine is not None:
        ay, ax = cy - spine[0], cx - spine[1]
        norm = np.hypot(ay, ax) + 1e-6
        anterior = (d_y * ay + d_x * ax) / norm
    else:
        anterior = np.zeros_like(d_y)
    mediastinal = (medial > cfg.costal_medial_cos) & (anterior > -cfg.costal_posterior_cos)

    margin = cfg.costal_margin_mm / px
    min_gap = cfg.costal_min_gap_mm / px
    stop = r_rib - margin
    valid = has_lung & (r_rib < r_max) & ((stop - r_l) >= min_gap) & ~mediastinal

    seg = (idx > r_l[:, None]) & (idx < stop[:, None]) & valid[:, None]
    n_seg = seg.sum(1)
    fat = seg & (hu_r > cfg.costal_fat_low) & (hu_r < cfg.costal_fat_high)
    fat_frac = fat.sum(1) / np.maximum(n_seg, 1)
    bad = (seg & (~body_r | other_r)).any(1) | (fat_frac > cfg.costal_max_fat_fraction)
    seg &= ~bad[:, None]

    added = np.zeros_like(side)
    added[yi[seg], xi[seg]] = True
    added = closing(added, 2)
    added &= body & ~bone & ~other & ~side & (inorm <= cfg.costal_dense_high)
    if not added.any():
        return added
    # keep only what is connected to the lung
    lab, n = label2d(added)
    touch = np.unique(lab[ndi.binary_dilation(side) & added])
    added = np.isin(lab, touch[touch > 0])
    return added


def costal_envelope_volume(inorm_vol, right_vol, left_vol, body_vol, spacing, cfg: Config):
    sz, sy, sx = spacing
    add_r = np.zeros(inorm_vol.shape, bool)
    add_l = np.zeros_like(add_r)
    from .alg3_separation import body_midline
    for z in range(inorm_vol.shape[0]):
        r, l = right_vol[z], left_vol[z]
        if not (r.any() or l.any()):
            continue
        body = body_vol[z]
        bone = remove_small((inorm_vol[z] > cfg.bone_threshold) & body, 10)
        mid = body_midline(body)
        spine = _spine_centroid(bone, mid, sx)
        add_r[z] = _envelope_side(inorm_vol[z], r, l, bone, body, mid, spine, (sy, sx), cfg)
        add_l[z] = _envelope_side(inorm_vol[z], l, r | add_r[z], bone, body, mid, spine, (sy, sx), cfg)

    struct = ndi.generate_binary_structure(3, 3)
    for add in (add_r, add_l):
        lab, n = ndi.label(add, structure=struct)
        for i, sl in enumerate(ndi.find_objects(lab), start=1):
            if sl is None:
                continue
            if sl[0].stop - sl[0].start < cfg.cavity_min_z_span or (lab[sl] == i).sum() < cfg.cavity_min_voxels:
                add[lab == i] = False

    right = right_vol | add_r
    left = left_vol | (add_l & ~right)
    for z in range(inorm_vol.shape[0]):
        right[z] = fill_holes(right[z])
        left[z] = fill_holes(left[z]) & ~right[z]
    return right, left, add_r, add_l
