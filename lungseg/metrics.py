"""Evaluation metrics (whole lung): overlap, surface distance, volume, and the
lesion inclusion rate, which directly measures whether diseased lung is kept."""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi


def _surface(mask):
    return mask & ~ndi.binary_erosion(mask, structure=ndi.generate_binary_structure(3, 1))


def surface_distances(pred, gt, spacing):
    """Symmetric surface distances (mm) between two binary volumes.
    Computed inside the joint bounding box (+ margin) to save memory."""
    both = pred | gt
    if both.any():
        idx = np.argwhere(both)
        lo = np.maximum(idx.min(0) - 3, 0)
        hi = np.minimum(idx.max(0) + 4, both.shape)
        sl = tuple(slice(a, b) for a, b in zip(lo, hi))
        pred, gt = pred[sl], gt[sl]
    sp, sg = _surface(pred), _surface(gt)
    if not sp.any() or not sg.any():
        return np.array([np.inf])
    dt_g = ndi.distance_transform_edt(~sg, sampling=spacing)
    dt_p = ndi.distance_transform_edt(~sp, sampling=spacing)
    return np.concatenate([dt_g[sp], dt_p[sg]])


def overlap_metrics(pred, gt, spacing):
    pred, gt = pred.astype(bool), gt.astype(bool)
    tp = np.logical_and(pred, gt).sum()
    fp = np.logical_and(pred, ~gt).sum()
    fn = np.logical_and(~pred, gt).sum()
    vox = float(np.prod(spacing)) / 1000.0  # mL
    d = surface_distances(pred, gt, spacing)
    vp, vg = pred.sum() * vox, gt.sum() * vox
    return {
        "dice": 2 * tp / max(2 * tp + fp + fn, 1),
        "iou": tp / max(tp + fp + fn, 1),
        "precision": tp / max(tp + fp, 1),
        "recall": tp / max(tp + fn, 1),
        "hd95_mm": float(np.percentile(d, 95)),
        "assd_mm": float(d.mean()),
        "vol_pred_ml": vp,
        "vol_gt_ml": vg,
        "signed_vol_err_pct": 100 * (vp - vg) / max(vg, 1e-6),
        "abs_vol_err_pct": 100 * abs(vp - vg) / max(vg, 1e-6),
    }


def lesion_inclusion(pred, lesion_labels, class_names=None):
    """Fraction of each lesion class contained in the predicted lung.

    MedSeg infection masks: 1 = ground-glass, 2 = consolidation, 3 = pleural
    effusion. Effusion is outside the lung by definition and is reported only
    as a false-positive check (lower is better).
    """
    class_names = class_names or {1: "ggo", 2: "consolidation", 3: "effusion"}
    pred = pred.astype(bool)
    out = {}
    parenchymal = [k for k in class_names if class_names[k] != "effusion"]
    any_lesion = np.isin(lesion_labels, parenchymal)
    out["lir_all_parenchymal"] = pred[any_lesion].mean() if any_lesion.any() else np.nan
    for k, name in class_names.items():
        m = lesion_labels == k
        out[f"lir_{name}"] = pred[m].mean() if m.any() else np.nan
    return out


def affected_slice_recall(pred, gt, lesion_labels, min_lesion_px=500):
    """Lung recall restricted to slices with substantial disease."""
    z = np.where((lesion_labels > 0).reshape(lesion_labels.shape[0], -1).sum(1) >= min_lesion_px)[0]
    if z.size == 0:
        return np.nan
    g = gt[z].astype(bool)
    return np.logical_and(pred[z].astype(bool), g).sum() / max(g.sum(), 1)


def affected_region_metrics(pred, gt, lesion_labels, spacing, margin_mm=10.0, min_lesion_px=50):
    """Lung segmentation quality where the disease is.

    dice_affected_slices : lung Dice restricted to slices containing infection
    dice_lesion_region   : lung Dice inside the infection mask dilated by
                           `margin_mm` in-plane (the diseased neighbourhood)
    """
    pred, gt = pred.astype(bool), gt.astype(bool)
    inf = lesion_labels > 0
    out = {"dice_affected_slices": np.nan, "dice_lesion_region": np.nan}
    zs = np.where(inf.reshape(inf.shape[0], -1).sum(1) >= min_lesion_px)[0]
    if zs.size == 0:
        return out
    p, g = pred[zs], gt[zs]
    out["dice_affected_slices"] = 2 * (p & g).sum() / max(p.sum() + g.sum(), 1)
    r = max(1, int(round(margin_mm / min(spacing[1:]))))
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
    st = np.zeros((1, 2 * r + 1, 2 * r + 1), bool)
    st[0] = yy ** 2 + xx ** 2 <= r * r
    roi = ndi.binary_dilation(inf[zs], structure=st)
    p, g = p & roi, g & roi
    out["dice_lesion_region"] = 2 * (p & g).sum() / max(p.sum() + g.sum(), 1)
    return out
