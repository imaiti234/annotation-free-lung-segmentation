"""
Central configuration for the adaptive-elbow lung segmentation pipeline.

Every tunable constant lives here so that (i) the paper can list all of them in a
single supplementary table and (ii) a run can be frozen by saving this object to
JSON next to its outputs. Do NOT tune these on the evaluation data; freeze them
before running the external (MedSeg) evaluation.

Intensity convention
--------------------
All intensity thresholds are on the pseudo-HU scale produced by Algorithm 1
(range -1000 ... +1000). For calibrated CT in Hounsfield units the scale equals
HU (values are only clipped), so the numbers below can be read as HU.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Optional, Tuple
import json


@dataclass
class Config:
    # Bumped whenever a default changes; a frozen config from another version is refused.
    config_version: str = "2026-10-15c_lesion_model"
    # ------------------------------------------------------------------ general
    n_jobs: int = -1                  # parallel slices in Algorithm 2 (-1 = all cores)
    random_seed: int = 0              # pwlf uses a stochastic optimiser
    min_lung_component_px: int = 10   # paper: area-based outlier rejection (<10 px)
    # Lung never touches the body surface; candidate regions that reach this band
    # inside the skin (subcutaneous fat, skin folds) are rejected. 0 = off.
    body_margin_px: int = 5

    # ----------------------------------------------------- Algorithm 1 (coarse)
    # "auto": calibrated HU data are clipped to [-1000, 1000] (paper: "if data is
    # already within the range, normalisation is not performed"); otherwise the
    # paper's tail-percentile normalisation (Eq. 1) is applied.
    normalisation: str = "auto"       # "auto" | "paper" | "none"
    tail_percent: float = 5.0         # top/bottom 5 % = bone / air (paper)
    n_min: float = -1000.0
    n_max: float = 1000.0
    coarse_threshold: float = -320.0  # body/non-body threshold for the coarse lung

    # ------------------------------------------------ Algorithm 2 (elbow method)
    # Threshold sweep: 61 values, as in the manuscript. The upper limit stays
    # below adipose tissue (-190 HU): above it, subcutaneous / mediastinal fat
    # enclosed by skin or soft tissue is mistaken for lung (seen on COVID-19-CT-Seg).
    # Denser lesions are recovered by Algorithms 4 and 4b, not by the threshold.
    sweep_min: float = -500.0
    sweep_max: float = -200.0
    sweep_step: float = 5.0
    min_segments: int = 2             # piece-wise linear fit orders tried
    max_segments: int = 5
    # How K* is chosen. The manuscript says "minimum LSE", but the LSE of a
    # piece-wise fit never increases with K, so that rule always returns
    # max_segments. "bic" is the penalised version of the same idea.
    # "fixed2" reproduces the original code (2 segments, single breakpoint).
    segment_selection: str = "bic"    # "bic" | "fixed2" | "min_lse"
    pwlf_pop: int = 3                 # multi-start count for pwlf.fitfast
    # Paper: "the threshold value just before the elbow point".
    threshold_rounding: str = "before"  # "before" | "nearest"
    min_lung_area_for_elbow_px: int = 200  # slices with less lung skip the fit
    elbow_downsample: int = 2         # area curve computed on a 2x downsampled slice (speed only)
    threshold_median_filter: int = 0  # 0 = off (paper). e.g. 5 = smooth Th* along z

    # ------------------------------------------- Algorithm 3 (lung separation)
    separation_ratio: float = 0.3     # R = area(L2) / area(L1)
    separation_step: float = 5.0      # Th2 = Th2 - 5
    separation_min_threshold: float = -950.0
    bridge_dilation_iterations: int = 5
    bridge_kernel_size: int = 5
    midline_split_fallback: bool = True  # split along body midline if still joined

    # -------------------------------------------- Algorithm 4 (adaptive merging)
    lesion_low: float = -850.0        # admissible lesion intensities
    lesion_high: float = -150.0
    min_internal_lesion_px: int = 50  # minimum internal-lesion pixels for slice stats
    candidate_std_factor: float = 2.0 # pixel candidates: mL +/- k*stdL
    merge_mean_tolerance: float = 40.0  # paper: (mL-40) < mC < (mL+40)
    merge_distance_px: float = 15.0   # paper: min dist(c, lung boundary) < 15 px
    merge_iterations: int = 15        # paper: repeat 15 times
    closing_radius_px: int = 2
    # Selective hole filling (A4, A4b): fill holes <= this area or with lesion
    # density (<= lesion_high). Large dense holes (liver dome, heart) stay open.
    hole_max_area_mm2: float = 300.0
    fill_holes_3d: bool = True        # fill holes enclosed by lung in 3D (consolidation islands)

    # ---------------- Algorithm 4b (NEW): thoracic-cavity-constrained recovery
    enable_cavity_recovery: bool = True
    bone_threshold: float = 200.0     # ribs / spine / sternum
    rib_search_mm: float = 40.0       # bones within this distance of the lung define the cavity
    spine_halfwidth_mm: float = 25.0  # bone this close to the midline belongs to both sides
    cavity_low: float = -900.0        # recovered tissue: diseased parenchyma (GGO / crazy paving
    cavity_high: float = -150.0       #   / partial consolidation). Excludes fat, muscle, heart.
    cavity_opening_mm: float = 2.0    # removes thin partial-volume rims from the candidates
    cavity_closing_mm: float = 3.0    # closes dense consolidation islands inside recovered region
    cavity_min_z_span: int = 2        # 3D support: recovered blob must span >= 2 slices
    cavity_min_voxels: int = 50
    cavity_max_growth: float = 2.5    # safety: side area after recovery <= 2.5 x before
    # Optional dense consolidation wedged against the chest wall. Validate on the
    # development data (check the heart is never added) before switching on.
    enable_dense_pleural_recovery: bool = False
    dense_high: float = 100.0
    pleural_band_mm: float = 10.0
    pleural_contact_fraction: float = 0.5

    # ------------- Algorithm 4d (NEW): patient-specific lesion model + geodesic growing
    enable_lesion_model: bool = True
    texture_window_mm: float = 4.0    # local mean / local SD window
    lesion_sample_min: float = -700.0 # lung pixels denser than this = lesion/vessel class
    deep_mediastinum_mm: float = 15.0 # soft tissue this far from lung = mediastinum/heart class
    llr_threshold: float = 0.0        # add if lesion-likelihood > soft-tissue likelihood
    max_grow_mm: float = 30.0         # geodesic distance limit from the lung boundary

    # ------------- Algorithm 4c (NEW): rib-cage radial envelope (dense consolidation)
    enable_costal_envelope: bool = False  # experimental; off (failed development test)
    costal_angle_step_deg: float = 0.5
    costal_window_deg: float = 6.0     # bridges intercostal spaces (inner rib surface)
    costal_max_gap_mm: float = 60.0    # thickest consolidation layer considered
    costal_min_gap_mm: float = 4.0     # thinner gaps are normal pleura / chest wall
    costal_margin_mm: float = 3.0      # stay this far inside the rib surface
    costal_fat_low: float = -190.0     # adipose tissue range
    costal_fat_high: float = -60.0
    costal_max_fat_fraction: float = 0.15
    costal_medial_cos: float = 0.35    # rays pointing medially ...
    costal_posterior_cos: float = 0.30 # ... and not clearly posterior = mediastinum (excluded)
    costal_dense_high: float = 150.0   # densest tissue that may be added

    # -------------------- Algorithm 5b (NEW): trachea / main-bronchi removal
    enable_airway_removal: bool = True
    airway_threshold: float = -950.0  # region-growing threshold (relaxed up to the seed threshold)
    airway_seed_threshold: float = -900.0
    airway_min_ml: float = 20.0       # trachea + main bronchi are at least this large
    airway_max_ml: float = 250.0      # larger grown volume = leak into lung
    airway_min_radius_mm: float = 3.0 # keep only large-calibre airways
    airway_wall_px: int = 1

    # --------------------------------------- Algorithm 5 (3D connectivity)
    abrupt_change_fraction: float = 0.10   # paper: area(LD) > 10 % of area(pB)
    correction_overlap_fraction: float = 0.95  # paper: area(LC) > 95 % of area(LD)
    remove_small_3d_fraction: float = 0.05  # drop 3D fragments < 5 % of the largest (0 = off)

    def to_json(self, path) -> None:
        with open(path, "w") as fh:
            json.dump(asdict(self), fh, indent=2)

    @classmethod
    def from_json(cls, path) -> "Config":
        with open(path) as fh:
            return cls(**json.load(fh))

    def ablation(self, stage: str) -> "Config":
        """Return a copy with one optional stage switched off (for ablations)."""
        d = asdict(self)
        if stage == "no_cavity":
            d["enable_cavity_recovery"] = False
        else:
            raise ValueError(stage)
        return Config(**d)
