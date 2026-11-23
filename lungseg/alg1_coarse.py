"""
Algorithm 1 - Coarse lung segmentation (manuscript Section 4.1).

1-3. Normalise intensities to [-1000, 1000] (Eq. 1) using the mean of the
     darkest 5 % (air) and brightest 5 % (bone) pixels after rejecting outlier
     components < 10 px.
4.   Threshold: body/tissue = Inorm > Th.
5.   B1 = largest white region (the body).
6.   B2 = fill holes of B1 (complete foreground).
7.   Lung = XOR(B1, B2)  (Eq. 2): everything enclosed by the body that is darker
     than Th.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage as ndi

from .config import Config
from .morph import STRUCT2D, remove_small, largest_k, fill_holes


def _paper_normalise_slice(img: np.ndarray, cfg: Config) -> np.ndarray:
    lo = np.percentile(img, cfg.tail_percent)
    hi = np.percentile(img, 100.0 - cfg.tail_percent)
    air = remove_small(img <= lo, cfg.min_lung_component_px)
    bone = remove_small(img >= hi, cfg.min_lung_component_px)
    img_min = float(img[air].mean()) if air.any() else float(lo)
    img_max = float(img[bone].mean()) if bone.any() else float(hi)
    if img_max - img_min < 1e-6:
        return np.full_like(img, cfg.n_min, dtype=np.float32)
    out = cfg.n_min + (img - img_min) / (img_max - img_min) * (cfg.n_max - cfg.n_min)
    return np.clip(out, cfg.n_min, cfg.n_max).astype(np.float32)


def looks_calibrated_hu(vol: np.ndarray) -> bool:
    p1, p99 = np.percentile(vol[:: max(1, vol.shape[0] // 20)], [1, 99])
    return p1 < -850 and -400 < p99 < 4000


def normalise_volume(vol: np.ndarray, cfg: Config) -> np.ndarray:
    """Eq. (1). Returns the pseudo-HU volume Inorm (float32, Z,Y,X)."""
    mode = cfg.normalisation
    if mode == "auto":
        mode = "none" if looks_calibrated_hu(vol) else "paper"
    if mode == "none":
        return np.clip(vol, cfg.n_min, cfg.n_max).astype(np.float32)
    if mode == "paper":
        return np.stack([_paper_normalise_slice(s, cfg) for s in vol]).astype(np.float32)
    raise ValueError(f"unknown normalisation mode {cfg.normalisation}")


@dataclass
class SliceThresholdResult:
    lung: np.ndarray       # XOR(B1, B2) after outlier rejection (all candidate regions)
    body: np.ndarray       # B1
    body_filled: np.ndarray  # B2


def segment_at_threshold(inorm: np.ndarray, th: float, cfg: Config) -> SliceThresholdResult:
    """Steps 4-7 of Algorithm 1 for one slice and one threshold."""
    white = remove_small(inorm > th, cfg.min_lung_component_px)
    b1 = largest_k(white, 1)
    b2 = fill_holes(b1)
    lung = b2 & ~b1
    lung = remove_small(lung, cfg.min_lung_component_px)
    if cfg.body_margin_px > 0 and lung.any():
        # reject regions touching the band just inside the body surface
        inner = ndi.binary_erosion(b2, iterations=cfg.body_margin_px)
        band = b2 & ~inner
        lab, n = ndi.label(lung, structure=STRUCT2D)
        touching = np.unique(lab[band & lung])
        touching = touching[touching > 0]
        if touching.size:
            lung &= ~np.isin(lab, touching)
    return SliceThresholdResult(lung=lung, body=b1, body_filled=b2)


def coarse_volume(inorm_vol: np.ndarray, cfg: Config):
    """Coarse lung (two largest regions) and body mask for every slice."""
    coarse = np.zeros(inorm_vol.shape, bool)
    body = np.zeros(inorm_vol.shape, bool)
    for z, s in enumerate(inorm_vol):
        r = segment_at_threshold(s, cfg.coarse_threshold, cfg)
        coarse[z] = largest_k(r.lung, 2)
        body[z] = r.body_filled
    return coarse, body
