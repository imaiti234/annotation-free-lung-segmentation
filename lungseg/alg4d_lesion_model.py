"""
Algorithm 4d (NEW) - Patient-specific lesion model and lesion-guided growing.

Generalises Algorithm 4 of the manuscript ("lesion characteristics + distance
from the lung") from a single mean-intensity rule to a two-class likelihood
model learned from the SAME scan (no training data, no ground truth):

1. Features per pixel (2D, per slice): local mean HU and log local standard
   deviation in a small window. Diseased lung (GGO, crazy paving, consolidation
   with air bronchograms) is heterogeneous; muscle, heart, liver and effusion
   are homogeneous.
2. Lesion class  L: pixels of the current lung mask with density above
   `lesion_sample_min` (diseased parenchyma and vessels already accepted as lung).
   Soft-tissue class B: (a) chest-wall muscle outside the thoracic hull and
   (b) deep mediastinum (> `deep_mediastinum_mm` from the lung).
   Each class is a 2D Gaussian (mean, covariance) fitted per patient.
3. Candidates: inside the thoracic hull (lung + nearby ribs/vertebra), not bone,
   density in [-900, +150] HU, log-likelihood ratio LLR = log p(x|L) - log p(x|B)
   > `llr_threshold`.
4. Geodesic growing from the lung boundary through candidate pixels, limited to
   `max_grow_mm` (the "distance from the lung" criterion of Algorithm 4),
   followed by an opening (removes partial-volume rims) and 3D persistence
   (every added blob must span >= 2 slices).
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi
from skimage.morphology import disk

from .alg3_separation import body_midline
from .alg4b_cavity_recovery import convex_hull_image
from .config import Config
from .morph import fill_holes_selective, remove_small


def _features(slice_hu, win):
    m = ndi.uniform_filter(slice_hu, win)
    m2 = ndi.uniform_filter(slice_hu * slice_hu, win)
    sd = np.sqrt(np.maximum(m2 - m * m, 1.0))
    return m, np.log(sd)


class _Gauss:
    def __init__(self, x):
        self.mu = x.mean(0)
        self.cov = np.cov(x.T) + np.eye(2) * 1e-3
        self.icov = np.linalg.inv(self.cov)
        self.logdet = np.log(np.linalg.det(self.cov))

    def logpdf(self, x):
        d = x - self.mu
        return -0.5 * (np.einsum("ij,jk,ik->i", d, self.icov, d) + self.logdet)


def _thoracic_hull(lung, bone, spacing_yx, cfg):
    if not lung.any():
        return np.zeros_like(lung)
    dist = ndi.distance_transform_edt(~lung, sampling=spacing_yx)
    near = bone & (dist <= cfg.rib_search_mm)
    return convex_hull_image(lung | near)


def lesion_model_volume(inorm, right, left, body, spacing, cfg: Config):
    sz, sy, sx = spacing
    win = max(3, int(round(cfg.texture_window_mm / min(sy, sx))) | 1)
    nz = inorm.shape[0]
    lung = right | left
    zs = np.where(lung.reshape(nz, -1).any(1))[0]
    if zs.size == 0:
        return right, left, np.zeros_like(right)

    # ---- pass 1: collect class samples from the whole scan --------------------
    rng = np.random.default_rng(0)
    les_s, bg_s, hulls, bones, feats = [], [], {}, {}, {}
    for z in zs:
        hu = inorm[z]
        bone = remove_small((hu > cfg.bone_threshold) & body[z], 10)
        hull = _thoracic_hull(lung[z], bone, (sy, sx), cfg)
        m, ls = _features(hu, win)
        hulls[z], bones[z], feats[z] = hull, bone, (m, ls)
        les = lung[z] & (hu > cfg.lesion_sample_min) & (hu < 150)
        muscle = body[z] & ~ndi.binary_dilation(hull, iterations=3) & (hu > -50) & (hu < 150)
        deep = (ndi.distance_transform_edt(~lung[z], sampling=(sy, sx)) > cfg.deep_mediastinum_mm) \
            & hull & (hu > -50) & (hu < 150) & ~bone
        for mask, store in ((les, les_s), (muscle | deep, bg_s)):
            idx = np.flatnonzero(mask)
            if idx.size:
                idx = rng.choice(idx, min(idx.size, 400), replace=False)
                store.append(np.stack([m.ravel()[idx], ls.ravel()[idx]], 1))
    if not les_s or not bg_s:
        return right, left, np.zeros_like(right)
    L, B = np.concatenate(les_s), np.concatenate(bg_s)
    if len(L) < 200 or len(B) < 200:
        return right, left, np.zeros_like(right)
    gl, gb = _Gauss(L), _Gauss(B)

    # ---- pass 2: lesion-guided geodesic growing ---------------------------------
    r_open = max(1, int(round(cfg.cavity_opening_mm / min(sy, sx))))
    n_iter = max(1, int(round(cfg.max_grow_mm / min(sy, sx))))
    max_px = cfg.hole_max_area_mm2 / (sy * sx)
    add = np.zeros_like(lung)
    for z in zs:
        hu = inorm[z]
        m, ls = feats[z]
        x = np.stack([m.ravel(), ls.ravel()], 1)
        llr = (gl.logpdf(x) - gb.logpdf(x)).reshape(hu.shape)
        cand = hulls[z] & body[z] & ~lung[z] & ~bones[z] & (hu > -900) & (hu < 150) \
            & (llr > cfg.llr_threshold)
        if not cand.any():
            continue
        cand = ndi.binary_opening(cand, structure=disk(r_open))
        grown = ndi.binary_dilation(lung[z], iterations=n_iter, mask=cand | lung[z])
        add[z] = grown & ~lung[z]

    # 3D persistence
    struct = ndi.generate_binary_structure(3, 3)
    lab, n = ndi.label(add, structure=struct)
    for i, sl in enumerate(ndi.find_objects(lab), start=1):
        if sl is None:
            continue
        if sl[0].stop - sl[0].start < cfg.cavity_min_z_span or (lab[sl] == i).sum() < cfg.cavity_min_voxels:
            add[lab == i] = False

    # assign additions to the nearer lung
    r_out, l_out = right.copy(), left.copy()
    for z in np.where(add.reshape(nz, -1).any(1))[0]:
        dr = ndi.distance_transform_edt(~right[z]) if right[z].any() else np.full(add[z].shape, np.inf)
        dl = ndi.distance_transform_edt(~left[z]) if left[z].any() else np.full(add[z].shape, np.inf)
        r_out[z] |= add[z] & (dr <= dl)
        l_out[z] |= add[z] & (dl < dr)
        r_out[z] = fill_holes_selective(r_out[z], inorm[z], max_px, cfg.lesion_high)
        l_out[z] = fill_holes_selective(l_out[z], inorm[z], max_px, cfg.lesion_high) & ~r_out[z]
    return r_out, l_out, add
