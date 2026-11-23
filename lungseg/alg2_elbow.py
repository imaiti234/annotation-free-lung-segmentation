"""
Algorithm 2 - Adaptive thresholding by the elbow method (manuscript Section 4.2).

For every slice:
 1-2. Sweep Th1 over the threshold range; run Algorithm 1 steps 4-7 for each value
      and record the lung area.
 3-6. Fit continuous piece-wise linear functions (pwlf) with k segments and select
      the order K*.
 7-8. Among the breakpoints of the K* fit, pick the one with the maximum change of
      slope (the elbow); Th1* is the sweep value just before it.
 9.   Lung_refined2 = Algorithm 1 at Th1*; BW4reg = four largest regions.
 10.  Lung_refined3 = two largest regions.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import numpy as np
import pwlf
from joblib import Parallel, delayed
from scipy.ndimage import median_filter

from .alg1_coarse import segment_at_threshold
from .config import Config
from .morph import largest_k


@dataclass
class ElbowResult:
    threshold: np.ndarray          # Th1* per slice (np.nan where not fitted)
    fitted: np.ndarray             # bool per slice
    n_segments: np.ndarray         # K* per slice
    sweep: np.ndarray              # sweep values
    areas: np.ndarray              # (Z, n_sweep) lung area curves
    refined2: np.ndarray           # all lung regions at Th1*
    bw4reg: np.ndarray             # four largest regions
    refined3: np.ndarray           # two largest regions


def sweep_values(cfg: Config) -> np.ndarray:
    n = int(round((cfg.sweep_max - cfg.sweep_min) / cfg.sweep_step)) + 1
    return cfg.sweep_min + cfg.sweep_step * np.arange(n)


def _fit_order(x, y, k, cfg, seed):
    model = pwlf.PiecewiseLinFit(x, y, seed=seed)
    breaks = model.fitfast(k, pop=cfg.pwlf_pop)
    ssr = float(model.ssr)
    return model, np.asarray(breaks), ssr


def select_elbow(x: np.ndarray, area: np.ndarray, cfg: Config, seed: int):
    """Return (breakpoint, K*) for one area-vs-threshold curve."""
    y = area / max(area.max(), 1.0)
    n = len(x)
    if cfg.segment_selection == "fixed2":
        _, breaks, _ = _fit_order(x, y, 2, cfg, seed)
        return float(breaks[1]), 2

    best = None
    for k in range(cfg.min_segments, cfg.max_segments + 1):
        if n < 2 * k + 2:
            break
        model, breaks, ssr = _fit_order(x, y, k, cfg, seed)
        if cfg.segment_selection == "bic":
            score = n * np.log(ssr / n + 1e-12) + 2 * k * np.log(n)
        elif cfg.segment_selection == "min_lse":
            score = ssr
        else:
            raise ValueError(cfg.segment_selection)
        if best is None or score < best[0]:
            best = (score, k, model, breaks)

    _, k_star, model, breaks = best
    slopes = np.asarray(model.slopes)
    interior = breaks[1:-1]
    change = np.abs(np.diff(slopes))
    return float(interior[int(np.argmax(change))]), k_star


def _round_threshold(brk: float, sweep: np.ndarray, mode: str) -> float:
    if mode == "nearest":
        return float(sweep[np.argmin(np.abs(sweep - brk))])
    below = sweep[sweep <= brk]
    return float(below.max()) if below.size else float(sweep.min())


def _slice_job(inorm_slice: np.ndarray, sweep: np.ndarray, cfg: Config, seed: int):
    # The area curve only needs the SHAPE of area-vs-threshold, so it is computed
    # on a downsampled slice (block mean) for speed; the final mask uses full resolution.
    f = int(cfg.elbow_downsample)
    if f > 1:
        h, w = (inorm_slice.shape[0] // f) * f, (inorm_slice.shape[1] // f) * f
        inorm_slice = inorm_slice[:h, :w].reshape(h // f, f, w // f, f).mean(axis=(1, 3))
    areas = np.array(
        [segment_at_threshold(inorm_slice, th, cfg).lung.sum() for th in sweep],
        dtype=np.float64,
    )
    if areas.max() < cfg.min_lung_area_for_elbow_px:
        return areas, np.nan, 0
    brk, k = select_elbow(sweep, areas, cfg, seed)
    return areas, _round_threshold(brk, sweep, cfg.threshold_rounding), k


def adaptive_elbow_volume(inorm_vol: np.ndarray, cfg: Config) -> ElbowResult:
    sweep = sweep_values(cfg)
    jobs = Parallel(n_jobs=cfg.n_jobs)(
        delayed(_slice_job)(inorm_vol[z], sweep, cfg, cfg.random_seed + z)
        for z in range(inorm_vol.shape[0])
    )
    areas = np.stack([j[0] for j in jobs])
    th = np.array([j[1] for j in jobs], dtype=np.float64)
    ks = np.array([j[2] for j in jobs], dtype=np.int16)
    fitted = ~np.isnan(th)

    if cfg.threshold_median_filter and fitted.sum() > cfg.threshold_median_filter:
        idx = np.where(fitted)[0]
        th[idx] = median_filter(th[idx], size=cfg.threshold_median_filter, mode="nearest")

    th_used = np.where(fitted, th, cfg.coarse_threshold)
    refined2 = np.zeros(inorm_vol.shape, bool)
    bw4 = np.zeros_like(refined2)
    refined3 = np.zeros_like(refined2)
    for z in range(inorm_vol.shape[0]):
        lung = segment_at_threshold(inorm_vol[z], th_used[z], cfg).lung
        refined2[z] = lung
        bw4[z] = largest_k(lung, 4)
        refined3[z] = largest_k(lung, 2)
    return ElbowResult(th_used, fitted, ks, sweep, areas, refined2, bw4, refined3)
