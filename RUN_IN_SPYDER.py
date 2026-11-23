# -*- coding: utf-8 -*-
"""
RUN_IN_SPYDER.py  -  one-file runner for the adaptive-elbow pipeline
====================================================================

How to use (Spyder):
  1. Open this file in Spyder.
  2. Edit ONLY the "USER SETTINGS" block below.
  3. Press F5 (Run file).

First time: keep RUN_PHANTOM_TEST = True. It checks the installation on a
synthetic CT in about a minute. Then set it to False, set MAX_CASES = 1 for
one real case, and finally MAX_CASES = None for all cases.

Pipeline per case:  A1 coarse -> A2 adaptive elbow -> A3 lung separation ->
A4 adaptive merging -> A4b cavity-constrained recovery (new) -> A5 3D
connectivity. Every stage is saved, so the ablation table comes from one run.

The ground-truth masks are read only AFTER segmentation, for evaluation.
"""
from __future__ import annotations

import runpy
import sys
import time
import traceback
from pathlib import Path

# =============================================================================
# USER SETTINGS  (edit here)
# =============================================================================
_HERE = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
# COVID-19-CT-Seg (https://doi.org/10.5281/zenodo.3757476), unpacked into data/COVID19_CT_Seg
DATA_ROOT = _HERE / "data" / "COVID19_CT_Seg"
CT_DIR = DATA_ROOT / "all"                 # CT volumes (.nii / .nii.gz)
GT_LUNG_DIR = DATA_ROOT / "lung_mask"      # lung masks (evaluation only)
GT_LESION_DIR = DATA_ROOT / "Infection_Mask"   # optional (evaluation only); ignored if missing.
# COVID-19-CT-Seg ships "Infection_Mask" (1 = infection) and "Lung_and_Infection_Mask"
# (3 = infection); both work.
OUTPUT_ROOT = _HERE / "outputs" / "COVID19_ELBOW3D_OUTPUTS"  # everything is written here
# (the paper's run used configuration version "2026-10-15c_lesion_model";
#  its saved outputs are in results/covid19_ct_seg)

RUN_PHANTOM_TEST = False   # True = installation test on a synthetic CT, then stop
MAX_CASES = None              # 1 = first case only (smoke test); None = all cases
ONLY_CASES = None          # e.g. ["coronacases_001"] to run selected cases
OVERWRITE = False          # False = skip cases that are already segmented
SAVE_STAGES = True         # save A1...A5 masks (needed for the ablation)
N_JOBS = -1                # CPU cores for Algorithm 2 (-1 = all). Use 1 if you see
                           # multiprocessing errors in Spyder.

# Cases you have already looked at while developing/debugging the method.
# They are still segmented, but kept OUT of the external-validation statistics
# (report them separately, like LCTSC-Test-S3-104).
DEVELOPMENT_CASES = ["coronacases_001", "coronacases_002",
                     "radiopaedia_14_85914_0", "radiopaedia_27_86410_0"]
# ^ the four cases used to develop/debug the method (with Claude, 15 Oct 2026).
#   The other 16 cases are the untouched external test set.

RUN_SEGMENTATION = True
RUN_EVALUATION = True
RUN_STATISTICS = True
RUN_FIGURES = True
FIGURE_SLICES = 5          # rows in each QC figure

# COVID-19-CT-Seg infection masks are binary (1 = infection).
LESION_CLASS_NAMES = {1: "infection"}

# Radiopaedia volumes in COVID-19-CT-Seg can be stored windowed to 0..255 or 0..1.
# Same inverse mapping as in your A6 external-validation script (dataset adapter,
# not a method parameter).
RADIO_LOW_HU, RADIO_HIGH_HU = -1250.0, 250.0
# =============================================================================

PACKAGE_DIR = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
if str(PACKAGE_DIR) not in sys.path:
    sys.path.insert(0, str(PACKAGE_DIR))

import numpy as np                                    # noqa: E402
import pandas as pd                                   # noqa: E402
import matplotlib                                     # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                       # noqa: E402
from skimage import measure                           # noqa: E402

from lungseg.config import Config                     # noqa: E402
from lungseg.io_utils import load_volume, load_mask, to_display   # noqa: E402
from lungseg.metrics import (overlap_metrics, lesion_inclusion, affected_region_metrics,  # noqa: E402
                             affected_slice_recall)
from lungseg.pipeline import run_volume               # noqa: E402


# ----------------------------------------------------------------- file pairing
def nii_stem(p: Path) -> str:
    n = p.name
    return n[:-7] if n.lower().endswith(".nii.gz") else n[:-4] if n.lower().endswith(".nii") else p.stem


def pair_key(p: Path) -> str:
    k = nii_stem(p).lower().strip().replace("-", "_").replace(" ", "")
    k = k.replace("_org_", "_")
    if k.endswith("_org"):
        k = k[:-4]
    for suf in ("_lung_mask", "_lungmask", "_infection_mask", "_infection",
                "_whole_lung", "_mask", "_lung"):
        if k.endswith(suf):
            k = k[: -len(suf)]
            break
    while "__" in k:
        k = k.replace("__", "_")
    return k.strip("_")


def list_nii(folder: Path):
    if folder is None or not folder.exists():
        return []
    return sorted(p for p in folder.rglob("*") if p.is_file()
                  and p.name.lower().endswith((".nii", ".nii.gz")))


def build_cases():
    cts = list_nii(CT_DIR)
    if not cts:
        raise FileNotFoundError(f"No CT .nii/.nii.gz files found in {CT_DIR}")
    lung = {pair_key(p): p for p in list_nii(GT_LUNG_DIR)}
    les_dir = GT_LESION_DIR
    if les_dir is not None and not les_dir.exists():
        for alt in ("Infection_Mask", "infection_mask", "Lung_and_Infection_Mask"):
            if (DATA_ROOT / alt).exists():
                les_dir = DATA_ROOT / alt
                break
    lesion = {pair_key(p): p for p in list_nii(les_dir)}
    cases = []
    for ct in cts:
        k = pair_key(ct)
        cases.append({"case": nii_stem(ct), "ct": ct, "gt": lung.get(k), "lesion": lesion.get(k)})
    if ONLY_CASES:
        cases = [c for c in cases if c["case"] in set(ONLY_CASES)]
    if MAX_CASES is not None:
        cases = cases[: int(MAX_CASES)]
    return cases


# ------------------------------------------------------------ intensity adapter
def decode_to_hu(case_id: str, vol: np.ndarray):
    p1, mn, mx = np.percentile(vol, 1), float(vol.min()), float(vol.max())
    looks_hu = p1 < -500 or mn < -900
    if looks_hu:
        return vol.astype(np.float32), "native_HU"
    if "radio" in case_id.lower():
        if mn >= -0.05 and mx <= 1.05:
            return (vol * (RADIO_HIGH_HU - RADIO_LOW_HU) + RADIO_LOW_HU).astype(np.float32), "radio_0_1_to_HU"
        if mn >= -2 and mx <= 260:
            return (vol / 255.0 * (RADIO_HIGH_HU - RADIO_LOW_HU) + RADIO_LOW_HU).astype(np.float32), "radio_0_255_to_HU"
    raise RuntimeError(f"{case_id}: intensities are not HU-like (min={mn:.1f}, p1={p1:.1f}, "
                       f"max={mx:.1f}) and no known decoding applies.")


# ----------------------------------------------------------------- figures
def _outline(ax, mask, color, lw=1.2):
    for c in measure.find_contours(to_display(mask.astype(float)), 0.5):
        ax.plot(c[:, 1], c[:, 0], color=color, lw=lw)


def _show(ax, img, title):
    ax.imshow(to_display(img), cmap="gray", vmin=-1350, vmax=150)
    ax.set_title(title, fontsize=10)
    ax.axis("off")


def make_figures(case, hu, out_dir, fig_dir):
    gt = load_mask(case["gt"]) > 0 if case["gt"] else None
    les = load_mask(case["lesion"]) if case["lesion"] else None
    base = load_mask(out_dir / "A5_no_cavity.nii.gz") > 0
    final = load_mask(out_dir / "A5_final.nii.gz") > 0
    if les is not None and les.shape == hu.shape:
        score = (les > 0).reshape(les.shape[0], -1).sum(1)
    elif gt is not None and gt.shape == hu.shape:
        score = (gt & ~base).reshape(gt.shape[0], -1).sum(1)   # where the manuscript misses most
    else:
        score = final.reshape(final.shape[0], -1).sum(1)
    zs = sorted(int(z) for z in np.argsort(score)[::-1][:FIGURE_SLICES])

    ncol = 4 if gt is not None else 3
    fig, axes = plt.subplots(len(zs), ncol, figsize=(3.2 * ncol, 3.2 * len(zs)), squeeze=False)
    for r, z in enumerate(zs):
        c = 0
        _show(axes[r, c], hu[z], f"Raw CT | slice {z}"); c += 1
        if gt is not None:
            _show(axes[r, c], hu[z], "Ground truth")
            _outline(axes[r, c], gt[z], "#00e5ff")
            if les is not None:
                _outline(axes[r, c], les[z] > 0, "#ff3b30", 0.8)
            c += 1
        _show(axes[r, c], hu[z], "Without A4b"); _outline(axes[r, c], base[z], "#ff9f0a"); c += 1
        _show(axes[r, c], hu[z], "Proposed (final)"); _outline(axes[r, c], final[z], "#30d158")
    fig.tight_layout()
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_dir / f"{case['case']}_qc.png", dpi=170)
    plt.close(fig)

    curves = out_dir / "elbow_curves.npz"
    if curves.exists():
        d = np.load(curves)
        z = zs[len(zs) // 2]
        fig, ax = plt.subplots(figsize=(4.2, 3.2))
        ax.plot(d["sweep"], d["areas"][z], "o-", ms=3, color="#444")
        ax.axvline(d["threshold"][z], color="#ff3b30", ls="--", label=f"Th1* = {d['threshold'][z]:.0f} HU")
        ax.set_xlabel("Threshold (HU)"); ax.set_ylabel("Lung area (pixels)")
        ax.set_title(f"{case['case']} | elbow curve, slice {z}"); ax.legend(frameon=False)
        fig.tight_layout(); fig.savefig(fig_dir / f"{case['case']}_elbow_slice{z}.png", dpi=170)
        plt.close(fig)


# ----------------------------------------------------------------- phantom test
def phantom_test():
    print("=" * 80 + "\nINSTALLATION TEST ON A SYNTHETIC PHANTOM\n" + "=" * 80)
    mk = runpy.run_path(str(PACKAGE_DIR / "scripts" / "00_make_phantom.py"))
    ct, lung, les = mk["make"](seed=1)
    spacing = (2.5, 0.75, 0.75)
    affine = np.diag([0.75, 0.75, 2.5, 1.0])
    out = OUTPUT_ROOT / "_phantom_test"
    cfg = Config(n_jobs=N_JOBS)
    meta = run_volume(ct.astype(np.float32), spacing, affine, out, cfg)
    pred = load_mask(out / "A5_final.nii.gz") > 0
    m = overlap_metrics(pred, lung > 0, spacing)
    print(f"Phantom finished in {meta['timing_s']['total_s']:.1f} s | Dice = {m['dice']:.3f}")
    print("If this line is printed without errors, the installation works.")
    print("(The phantom is a software test, not a benchmark - ignore its Dice value.)")
    print("Now set RUN_PHANTOM_TEST = False and run again (F5).")


# ----------------------------------------------------------------- main
def main():
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    if RUN_PHANTOM_TEST:
        phantom_test()
        return

    cfg_path = OUTPUT_ROOT / "frozen_config.json"
    if cfg_path.exists():
        cfg = Config.from_json(cfg_path)
        if cfg.config_version != Config().config_version:
            raise RuntimeError(
                f"{cfg_path} was written by an older version of the code "
                f"({cfg.config_version!r}). Delete the whole folder {OUTPUT_ROOT} "
                "and run again so every case is segmented with the current method.")
        print(f"Using frozen configuration: {cfg_path}")
    else:
        cfg = Config()
        cfg.to_json(cfg_path)
        print(f"Created frozen configuration: {cfg_path}  (do not edit after evaluating)")
    cfg.n_jobs = N_JOBS

    cases = build_cases()
    print(f"{len(cases)} case(s) found in {CT_DIR}")
    for c in cases:
        print(f"  {c['case']:<35} GT={'yes' if c['gt'] else 'NO '}  lesion={'yes' if c['lesion'] else 'no'}")

    rows, failures = [], []
    for i, case in enumerate(cases, 1):
        cid = case["case"]
        out_dir = OUTPUT_ROOT / cid
        print(f"\n[{i}/{len(cases)}] {cid}")
        try:
            vol, spacing, affine = load_volume(case["ct"])
            hu, mode = decode_to_hu(cid, vol)
            print(f"  intensity: {mode} | shape {hu.shape} | spacing {tuple(round(s, 3) for s in spacing)} mm")

            if RUN_SEGMENTATION and (OVERWRITE or not (out_dir / "A5_final.nii.gz").exists()):
                meta = run_volume(hu, spacing, affine, out_dir, cfg, SAVE_STAGES, source=str(case["ct"]))
                print("  segmentation done: " + ", ".join(f"{k}={v:.1f}s" for k, v in meta["timing_s"].items()))
            else:
                print("  segmentation already exists - skipped")

            if RUN_EVALUATION and case["gt"]:
                gt = load_mask(case["gt"]) > 0
                les = load_mask(case["lesion"]) if case["lesion"] else None
                if les is not None and les.max() >= 3:      # Lung_and_Infection_Mask
                    les = (les == 3).astype(np.int16)
                elif les is not None:
                    les = (les > 0).astype(np.int16)
                for f in sorted(out_dir.glob("*.nii.gz")):
                    method = f.name.replace(".nii.gz", "")
                    pred = load_mask(f) > 0
                    if pred.shape != gt.shape:
                        print(f"  {method}: shape mismatch {pred.shape} vs GT {gt.shape} - skipped")
                        continue
                    row = {"case": cid, "method": method, "intensity_mode": mode}
                    row.update(overlap_metrics(pred, gt, spacing))
                    if les is not None and les.shape == gt.shape:
                        row.update(lesion_inclusion(pred, les, LESION_CLASS_NAMES))
                        row.update(affected_region_metrics(pred, gt, les, spacing))
                        row["recall_affected_slices"] = affected_slice_recall(pred, gt, les)
                    rows.append(row)
                fin = [r for r in rows if r["case"] == cid and r["method"] == "A5_final"]
                man = [r for r in rows if r["case"] == cid and r["method"] == "A5_no_cavity"]
                if fin and man:
                    print(f"  Dice  without-A4b={man[0]['dice']:.4f}  proposed={fin[0]['dice']:.4f}"
                          + (f" | infection inside lung  manuscript={man[0].get('lir_all_parenchymal', np.nan):.3f}"
                             f"  proposed={fin[0].get('lir_all_parenchymal', np.nan):.3f}" if les is not None else ""))

            if RUN_FIGURES and (out_dir / "A5_final.nii.gz").exists():
                make_figures(case, hu, out_dir, OUTPUT_ROOT / "_figures")
        except Exception as exc:
            print(f"  FAILED: {type(exc).__name__}: {exc}")
            failures.append({"case": cid, "error": f"{type(exc).__name__}: {exc}",
                             "traceback": traceback.format_exc()})

    rep = OUTPUT_ROOT / "_reports"
    rep.mkdir(exist_ok=True)
    if failures:
        pd.DataFrame(failures).to_csv(rep / "failed_cases.csv", index=False)
        print(f"\n{len(failures)} case(s) failed - see {rep / 'failed_cases.csv'}")
    if rows:
        metrics_csv = rep / "metrics_all_stages.csv"
        new = pd.DataFrame(rows)
        if metrics_csv.exists():   # keep results of cases evaluated in earlier runs
            old = pd.read_csv(metrics_csv)
            old = old[~old.case.isin(new.case.unique())]
            new = pd.concat([old, new], ignore_index=True)
        new.to_csv(metrics_csv, index=False)
        print(f"\nMetrics: {metrics_csv}")
        new["development_case"] = new.case.isin(DEVELOPMENT_CASES)
        new.to_csv(metrics_csv, index=False)
        external = new[~new.development_case]
        ext_csv = rep / "metrics_external_only.csv"
        external.to_csv(ext_csv, index=False)
        if DEVELOPMENT_CASES:
            print(f"Development cases excluded from statistics: {DEVELOPMENT_CASES}")
        if RUN_STATISTICS and external.case.nunique() >= 2:
            argv = sys.argv
            sys.argv = ["03_statistics.py", "--csv", str(ext_csv), "--out", str(rep / "stats")]
            try:
                runpy.run_path(str(PACKAGE_DIR / "scripts" / "03_statistics.py"), run_name="__main__")
            finally:
                sys.argv = argv
    print("\nDONE. Outputs in:", OUTPUT_ROOT)


if __name__ == "__main__":
    t0 = time.perf_counter()
    main()
    print(f"Total time {time.perf_counter() - t0:.1f} s")
