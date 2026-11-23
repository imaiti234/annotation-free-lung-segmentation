"""End-to-end pipeline: Algorithms 1 -> 2 -> 3 -> 4 -> 4b -> 4d -> 5 -> 5b.

Every stage is saved so the ablation table can be produced from one run.
Output labels: 1 = right lung, 2 = left lung (patient side, RAS+).
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .alg1_coarse import coarse_volume, normalise_volume
from .alg2_elbow import adaptive_elbow_volume
from .alg3_separation import separate_volume
from .alg4_merging import merge_volume
from .alg4b_cavity_recovery import cavity_recovery_volume
from .alg4c_costal_envelope import costal_envelope_volume
from .alg4d_lesion_model import lesion_model_volume
from .alg5_connectivity3d import connectivity_3d
from .alg5b_airway import remove_airways
from .config import Config
from .io_utils import load_volume, save_mask

STAGES = [
    "A1_coarse",
    "A2_elbow",
    "A3_separation",
    "A4_merging",
    "A5_no_cavity",   # manuscript pipeline (1-2-3-4-5) + airway removal, without 4b
    "A4b_cavity",
    "A4d_lesion_model",
    "A5_connectivity",
    "A5_final",       # proposed: 1-2-3-4-4b-(4c)-5-5b
]


def _labels(right, left):
    out = np.zeros(right.shape, np.uint8)
    out[right] = 1
    out[left & ~right] = 2
    return out


def run_case(image_path, out_dir, cfg: Config = None, save_stages: bool = True) -> dict:
    vol, spacing, affine = load_volume(image_path)
    return run_volume(vol, spacing, affine, out_dir, cfg, save_stages, source=str(image_path))


def run_volume(vol, spacing, affine, out_dir, cfg: Config = None, save_stages: bool = True,
               source: str = "") -> dict:
    """Run on an already-loaded (Z, Y, X) volume in HU (see io_utils.load_volume)."""
    cfg = cfg or Config()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    timing = {}
    t_all = time.perf_counter()

    t = time.perf_counter()
    inorm = normalise_volume(vol, cfg)
    coarse, body = coarse_volume(inorm, cfg)
    timing["A1"] = time.perf_counter() - t

    t = time.perf_counter()
    elbow = adaptive_elbow_volume(inorm, cfg)
    timing["A2"] = time.perf_counter() - t

    t = time.perf_counter()
    right3, left3, lowered = separate_volume(inorm, elbow.refined3, elbow.threshold, body, cfg)
    timing["A3"] = time.perf_counter() - t

    t = time.perf_counter()
    right4, left4, lesion_stats = merge_volume(inorm, right3, left3, body, cfg, spacing)
    timing["A4"] = time.perf_counter() - t

    r5n, l5n, _ = connectivity_3d(right4, left4, elbow.bw4reg, cfg)

    t = time.perf_counter()
    if cfg.enable_cavity_recovery:
        right4b, left4b, add_r, add_l = cavity_recovery_volume(inorm, right4, left4, body, spacing, cfg)
    else:
        right4b, left4b = right4, left4
    timing["A4b"] = time.perf_counter() - t

    t = time.perf_counter()
    if cfg.enable_lesion_model:
        right4d, left4d, _ = lesion_model_volume(inorm, right4b, left4b, body, spacing, cfg)
    else:
        right4d, left4d = right4b, left4b
    timing["A4d"] = time.perf_counter() - t

    t = time.perf_counter()
    if cfg.enable_costal_envelope:
        right4c, left4c, _, _ = costal_envelope_volume(inorm, right4d, left4d, body, spacing, cfg)
    else:
        right4c, left4c = right4d, left4d
    timing["A4c"] = time.perf_counter() - t

    t = time.perf_counter()
    right5, left5, restored = connectivity_3d(right4c, left4c, elbow.bw4reg, cfg)
    timing["A5"] = time.perf_counter() - t

    t = time.perf_counter()
    if cfg.enable_airway_removal:
        right5b, left5b, _ = remove_airways(inorm, right5, left5, body, spacing, cfg)
        r5n, l5n, _ = remove_airways(inorm, r5n, l5n, body, spacing, cfg)
    else:
        right5b, left5b = right5, left5
    timing["A5b"] = time.perf_counter() - t
    timing["total_s"] = time.perf_counter() - t_all

    stages = {
        "A1_coarse": coarse.astype(np.uint8),
        "A2_elbow": elbow.refined3.astype(np.uint8),
        "A3_separation": _labels(right3, left3),
        "A4_merging": _labels(right4, left4),
        "A5_no_cavity": _labels(r5n, l5n),
        "A4b_cavity": _labels(right4b, left4b),
        "A4d_lesion_model": _labels(right4d, left4d),
        "A5_connectivity": _labels(right5, left5),
        "A5_final": _labels(right5b, left5b),
    }
    for name, arr in stages.items():
        if save_stages or name == "A5_final":
            save_mask(arr, affine, out_dir / f"{name}.nii.gz")

    # per-slice diagnostics (elbow thresholds, curves) for figures / supplement
    diag = pd.DataFrame({
        "slice": np.arange(vol.shape[0]),
        "elbow_fitted": elbow.fitted,
        "threshold": elbow.threshold,
        "K_star": elbow.n_segments,
        "separation_lowered": lowered,
    })
    diag.to_csv(out_dir / "slice_diagnostics.csv", index=False)
    np.savez_compressed(out_dir / "elbow_curves.npz", sweep=elbow.sweep, areas=elbow.areas,
                        threshold=elbow.threshold)
    cfg.to_json(out_dir / "config_used.json")
    meta = {
        "image": source,
        "shape_zyx": list(vol.shape),
        "spacing_zyx_mm": list(spacing),
        "lesion_stats_volume": lesion_stats,
        "timing_s": timing,
    }
    with open(out_dir / "run_meta.json", "w") as fh:
        json.dump(meta, fh, indent=2)
    return meta
