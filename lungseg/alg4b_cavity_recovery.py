"""
Algorithm 4b (NEW) - Thoracic-cavity-constrained recovery of disease-obscured lung.

Why it is needed
----------------
Algorithms 1-4 define lung as tissue darker than an adaptive threshold. In
severely affected lungs (dense ground-glass, crazy paving, consolidation),
large lesions touch the chest wall and are brighter than Th1*, so they are
treated as "body". Algorithm 4 can only merge lesions whose centroid lies
within 15 px of the lung, which fails for large lesions.

Idea (keeps the manuscript's philosophy: model-free, intensity + anatomy)
---------------------------------------------------------------------------
The lung cannot extend beyond the rib cage. For each lung and slice:
 1. Bones (Inorm > 200) within `rib_search_mm` of the lung are found (ribs,
    vertebra, sternum). Bones near the midline are shared by both sides.
 2. The cavity envelope H = convex hull(lung u nearby bones), restricted to the
    body and excluding the other lung.
 3. Candidates = pixels in H, not bone, with diseased-parenchyma intensity
    (cavity_low .. cavity_high). The upper limit (-150) excludes fat, muscle,
    heart and vessels, which is what keeps the mediastinum out.
 4. Candidate components touching the lung are added; a small closing inside H
    plus hole filling absorbs dense consolidation islands surrounded by GGO.
 5. 3D support: each added 3D blob must span >= `cavity_min_z_span` slices
    (the 3D-connectivity principle of Algorithm 5 applied to additions).
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi
from scipy.spatial import ConvexHull, QhullError
from skimage.draw import polygon as draw_polygon
from skimage.morphology import disk

from .alg3_separation import body_midline
from .config import Config
from .morph import closing, fill_holes, fill_holes_selective, label2d, remove_small


def convex_hull_image(mask: np.ndarray) -> np.ndarray:
    """Fast convex hull of a binary mask (hull of boundary pixels, rasterised)."""
    out = np.zeros(mask.shape, bool)
    if mask.sum() < 3:
        return mask.copy()
    edge = mask & ~ndi.binary_erosion(mask)
    pts = np.argwhere(edge).astype(float)
    try:
        hull = ConvexHull(pts)
    except QhullError:
        return mask.copy()
    v = pts[hull.vertices]
    rr, cc = draw_polygon(v[:, 0], v[:, 1], mask.shape)
    out[rr, cc] = True
    return out | mask


def _recover_side(inorm, side, other, bone_near, side_cols, body, spacing_yx, cfg):
    if side.sum() < 20:
        return np.zeros_like(side)
    pts = side | (bone_near & side_cols)
    hull = convex_hull_image(pts) & body & ~other
    cand = hull & ~side & ~bone_near & (inorm >= cfg.cavity_low) & (inorm <= cfg.cavity_high)
    # Diseased parenchyma is a thick region; partial-volume rims along the heart,
    # diaphragm and chest wall are only 1-3 px wide. A morphological opening
    # (radius cavity_opening_mm) removes the rims and keeps real lesions.
    r_open = int(round(cfg.cavity_opening_mm / min(spacing_yx)))
    if r_open > 0 and cand.any():
        cand = ndi.binary_opening(cand, structure=disk(r_open))
    lab, n = label2d(cand)
    if n == 0:
        return np.zeros_like(side)
    touch = np.unique(lab[ndi.binary_dilation(side) & cand])
    touch = touch[touch > 0]
    rec = side | np.isin(lab, touch)
    r_px = int(round(cfg.cavity_closing_mm / min(spacing_yx)))
    max_px = cfg.hole_max_area_mm2 / (spacing_yx[0] * spacing_yx[1])
    rec = fill_holes_selective(closing(rec, r_px) & hull, inorm, max_px, cfg.lesion_high) & ~other
    rec &= ~(bone_near & ~side)
    if cfg.enable_dense_pleural_recovery and rec.any():
        rec |= _dense_pleural_pockets(inorm, rec, other, hull, bone_near, spacing_yx, cfg)
    if rec.sum() > cfg.cavity_max_growth * side.sum():
        return np.zeros_like(side)
    return rec & ~side


def _dense_pleural_pockets(inorm, lung, other, rib_hull, bone_near, spacing_yx, cfg):
    """Optional: dense consolidation wedged between the lung and the chest wall.

    A dense component (cavity_high .. dense_high) is added when it lies inside the
    convex hull of the (recovered) lung, touches the lung, and most of its contact
    with tissue outside that hull is close to ribs (i.e. it borders the chest
    wall, not the heart / mediastinum)."""
    lung_hull = convex_hull_image(lung) & rib_hull & ~other
    dense = lung_hull & ~lung & (inorm > cfg.cavity_high) & (inorm <= cfg.dense_high) & ~bone_near
    lab, n = label2d(dense)
    if n == 0:
        return np.zeros_like(lung)
    near_rib = ndi.distance_transform_edt(~bone_near, sampling=spacing_yx) <= cfg.pleural_band_mm \
        if bone_near.any() else np.zeros_like(lung)
    lung_ring = ndi.binary_dilation(lung)
    out = np.zeros_like(lung)
    for i in range(1, n + 1):
        cc = lab == i
        if not (cc & lung_ring).any():
            continue
        ring = ndi.binary_dilation(cc, iterations=2) & ~cc & ~lung & ~lung_hull
        if ring.sum() == 0 or near_rib[ring].mean() >= cfg.pleural_contact_fraction:
            out |= cc
    return out


def cavity_recovery_volume(inorm_vol, right_vol, left_vol, body_vol, spacing, cfg: Config):
    """Return (right, left, added_right, added_left)."""
    sz, sy, sx = spacing
    nz, ny, nx = inorm_vol.shape
    cols = np.arange(nx)[None, :]
    add_r = np.zeros(inorm_vol.shape, bool)
    add_l = np.zeros_like(add_r)

    for z in range(nz):
        r, l = right_vol[z], left_vol[z]
        lung = r | l
        if not lung.any():
            continue
        body = body_vol[z]
        bone = remove_small((inorm_vol[z] > cfg.bone_threshold) & body, 10)
        dist_mm = ndi.distance_transform_edt(~lung, sampling=(sy, sx))
        bone_near = bone & (dist_mm <= cfg.rib_search_mm)
        mid = body_midline(body)
        spine = np.abs(cols - mid) <= cfg.spine_halfwidth_mm / sx
        right_cols = np.broadcast_to((cols > mid) | spine, (ny, nx))
        left_cols = np.broadcast_to((cols <= mid) | spine, (ny, nx))
        add_r[z] = _recover_side(inorm_vol[z], r, l, bone_near, right_cols, body, (sy, sx), cfg)
        add_l[z] = _recover_side(inorm_vol[z], l, r | add_r[z], bone_near, left_cols, body, (sy, sx), cfg)

    struct = ndi.generate_binary_structure(3, 3)
    for add in (add_r, add_l):
        lab, n = ndi.label(add, structure=struct)
        if n == 0:
            continue
        for i, sl in enumerate(ndi.find_objects(lab), start=1):
            z_span = sl[0].stop - sl[0].start
            size = int((lab[sl] == i).sum())
            if z_span < cfg.cavity_min_z_span or size < cfg.cavity_min_voxels:
                add[lab == i] = False

    right = right_vol | add_r
    left = left_vol | (add_l & ~right)
    max_px = cfg.hole_max_area_mm2 / (sy * sx)
    for z in range(nz):
        right[z] = fill_holes_selective(right[z], inorm_vol[z], max_px, cfg.lesion_high)
        left[z] = fill_holes_selective(left[z], inorm_vol[z], max_px, cfg.lesion_high) & ~right[z]
    if cfg.fill_holes_3d:
        # regions enclosed by lung in 3D (e.g. a consolidation island) are lung
        right = ndi.binary_fill_holes(right)
        left = ndi.binary_fill_holes(left) & ~right
    return right, left, add_r, add_l & ~add_r
