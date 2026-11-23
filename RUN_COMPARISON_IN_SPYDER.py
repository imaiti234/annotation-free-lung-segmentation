# -*- coding: utf-8 -*-
"""
RUN_COMPARISON_IN_SPYDER.py - all comparators on COVID-19-CT-Seg, one file
==========================================================================

What it does (press F5; it can be stopped and restarted any time - finished
work is never repeated):
  1. Pretrained deep-learning comparators (CPU):
       lungmask R231, R231CovidWeb, LTRCLobes   (R231/R231CovidWeb are reused
                                                  if they already exist)
       TotalSegmentator v2 (nnU-Net), lung lobes merged to whole lung
  2. Classical non-learning comparators: Otsu + 3D components, Hu et al. 2001
  3. Evaluation of every method on the SAME cases, with the SAME reference
     masks and the SAME metric code; lesion-region metrics if the infection
     masks are present.
  4. Statistics on the 16 external cases (development cases excluded for
     every method), subgroup analysis by data source, run-time table, and a
     Dice box plot for the paper.

Install once (Spyder console, then restart the kernel):
    %pip install lungmask TotalSegmentator

Infection masks (evaluation only - needed for the "diseased lung" metrics):
    Download Infection_Mask.zip of the COVID-19 CT Lung and Infection
    Segmentation Dataset (Ma et al., Zenodo record 3757476) and extract it to
    <DATA_ROOT>\\Infection_Mask   (20 files named like the CT files).

The paths are taken from RUN_IN_SPYDER.py, so the two scripts always use the
same data, outputs and development-case list.
"""
from __future__ import annotations

import runpy
import sys
import time
from pathlib import Path

# =============================================================================
# USER SETTINGS
# =============================================================================
RUN_LUNGMASK = True
LUNGMASK_MODELS = ["R231", "R231CovidWeb", "LTRCLobes"]
RUN_TOTALSEGMENTATOR = True
TOTALSEG_FAST = False      # False = full-resolution 1.5 mm model (use this for the
                           # paper; ~3-10 min per case on CPU). True = 3 mm, quicker.
RUN_CLASSICAL = True
RUN_EVALUATION = True
RUN_STATISTICS = True
OVERWRITE = False
# =============================================================================

PACKAGE_DIR = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
if str(PACKAGE_DIR) not in sys.path:
    sys.path.insert(0, str(PACKAGE_DIR))

import numpy as np                                   # noqa: E402
import pandas as pd                                  # noqa: E402
import nibabel as nib                                # noqa: E402

from lungseg import comparison as C                  # noqa: E402

S = runpy.run_path(str(PACKAGE_DIR / "RUN_IN_SPYDER.py"), run_name="settings_only")
DATA_ROOT, OUTPUT_ROOT, CT_DIR = S["DATA_ROOT"], S["OUTPUT_ROOT"], S["CT_DIR"]
DEVELOPMENT_CASES = list(S["DEVELOPMENT_CASES"])

# Methods in the comparison table (order = table order).
METHODS = ["A5_final", "A5_no_cavity",
           "B_lungmask_R231", "B_lungmask_R231CovidWeb", "B_lungmask_LTRCLobes",
           "B_totalsegmentator", "C_otsu3d", "C_hu2001"]
SUBGROUPS = {"coronacases": "coronacases", "radiopaedia": "radiopaedia"}


def prepare_cases():
    g = S["build_cases"].__globals__        # always use ALL cases here
    g["ONLY_CASES"], g["MAX_CASES"] = None, None
    cases = S["build_cases"]()
    hu_dir = OUTPUT_ROOT / "_hu_input"
    for c in cases:
        img = nib.load(str(c["ct"]))
        out = hu_dir / f"{c['case']}.nii.gz"
        if out.exists():
            c["hu_path"] = out
            continue
        arr = np.asarray(img.dataobj, dtype=np.float32)
        hu, mode = S["decode_to_hu"](c["case"], arr)
        if mode == "native_HU":
            c["hu_path"] = c["ct"]            # already HU - use the original file
        else:                                 # identical decoding for every method
            hu_dir.mkdir(parents=True, exist_ok=True)
            nib.save(nib.Nifti1Image(np.rint(hu).astype(np.int16), img.affine), str(out))
            c["hu_path"] = out
            print(f"  {c['case']}: {mode} -> {out.name}")
    return cases


def statistics(df: pd.DataFrame, rep: Path):
    df = df.copy()
    df["development_case"] = df.case.isin(DEVELOPMENT_CASES)
    df.to_csv(rep / "comparison_metrics_all_cases.csv", index=False)
    ext = df[~df.development_case & df.method.isin(METHODS)]
    sets = {"stats_comparison": ext}
    for name, key in SUBGROUPS.items():
        sets[f"stats_comparison_{name}"] = ext[ext.case.str.contains(key)]
    for name, sub in sets.items():
        if sub.case.nunique() < 5:
            continue
        csv = rep / f"{name}.csv"
        sub.to_csv(csv, index=False)
        argv = sys.argv
        sys.argv = ["03_statistics.py", "--csv", str(csv), "--out", str(rep / name)]
        try:
            print(f"\n=== {name} (n = {sub.case.nunique()} cases) ===")
            runpy.run_path(str(PACKAGE_DIR / "scripts" / "03_statistics.py"), run_name="__main__")
        finally:
            sys.argv = argv
    C.dice_boxplot(ext, METHODS, rep / "figures" / "dice_boxplot_external.png",
                   f"COVID-19-CT-Seg, external cases (n = {ext.case.nunique()})")


def main():
    t0 = time.perf_counter()
    rep = OUTPUT_ROOT / "_reports"
    rep.mkdir(parents=True, exist_ok=True)
    cases = prepare_cases()
    n_les = sum(c["lesion"] is not None for c in cases)
    print(f"{len(cases)} cases | reference lung masks: {sum(c['gt'] is not None for c in cases)}"
          f" | infection masks: {n_les}")
    if n_les == 0:
        print("  NOTE: no infection masks found - lesion-region metrics will be empty.\n"
              f"  Put them in {DATA_ROOT / 'Infection_Mask'} (see the header of this file).")
    missing = [c["case"] for c in cases if not (OUTPUT_ROOT / c["case"] / "A5_final.nii.gz").exists()]
    if missing:
        print(f"  WARNING: proposed-method output missing for {missing}. Run RUN_IN_SPYDER.py first.")

    failures = []
    if RUN_CLASSICAL:
        print("\n--- classical comparators ---")
        C.run_classical(cases, OUTPUT_ROOT, OVERWRITE, failures)
    if RUN_LUNGMASK:
        print("\n--- lungmask ---")
        try:
            C.run_lungmask(cases, OUTPUT_ROOT, LUNGMASK_MODELS, OVERWRITE, failures)
        except ImportError:
            print("  lungmask is not installed:  %pip install lungmask")
    if RUN_TOTALSEGMENTATOR:
        print("\n--- TotalSegmentator ---")
        try:
            C.run_totalsegmentator(cases, OUTPUT_ROOT, TOTALSEG_FAST, OVERWRITE, failures)
        except ImportError:
            print("  TotalSegmentator is not installed:  %pip install TotalSegmentator")
    if failures:
        pd.DataFrame(failures).to_csv(rep / "comparison_failures.csv", index=False)
        print(f"\n{len(failures)} failure(s) - see {rep / 'comparison_failures.csv'}")

    if RUN_EVALUATION:
        print("\n--- evaluation (cached; only new/changed predictions are computed) ---")
        df = C.evaluate([c for c in cases if c["gt"]], OUTPUT_ROOT, METHODS,
                        rep / "comparison_metrics_cache.csv")
        rt = C.runtime_table(OUTPUT_ROOT, cases, METHODS)
        if not rt.empty:
            rt.to_csv(rep / "runtime_cpu_s.csv", index=False)
            print("\nCPU time per case (s):\n" + rt.round(1).to_string(index=False))
        if RUN_STATISTICS and not df.empty:
            statistics(df, rep)
    print(f"\nDONE in {(time.perf_counter() - t0) / 60:.1f} min. Results: {rep}")


if __name__ == "__main__":
    main()
