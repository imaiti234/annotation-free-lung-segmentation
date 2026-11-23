# -*- coding: utf-8 -*-
"""
RUN_LCTSC_IN_SPYDER.py - second dataset (LCTSC, 60 thoracic CTs), frozen method
================================================================================

Runs the SAME frozen pipeline as on COVID-19-CT-Seg (parameters are read from
the COVID run's frozen_config.json - nothing is re-tuned on LCTSC) and the SAME
comparators, then evaluates all of them on the SAME subjects against the SAME
reference masks.

Inputs:
  * LCTSC DICOM (CT series + RTSTRUCT), downloaded from The Cancer Imaging Archive
    (https://doi.org/10.7937/K9/TCIA.2017.3r3fvz08) into data/LCTSC.
  * Reference whole-lung masks are rasterised from the RTSTRUCT files
    (left + right lung) with lungseg/lctsc_io.py and cached in
    <OUTPUT_ROOT>/_reference. They are used for evaluation only.

LCTSC-Test-S3-104 was used to develop the original elbow method and is excluded
from the statistics for every method (as in the original LCTSC analysis).

Run RUN_IN_SPYDER.py (COVID-19-CT-Seg) first: this script reuses its
frozen_config.json, so nothing is re-tuned on LCTSC.

Long run (all methods on 60 subjects can take a night on CPU). It is safe to
stop and restart: finished work is skipped. Press F5.

Needs: pydicom, rt-utils, SimpleITK, lungmask, TotalSegmentator.
"""
from __future__ import annotations

import gzip
import json
import os
import runpy
import shutil
import sys
import time
import zlib
from pathlib import Path

# =============================================================================
# USER SETTINGS
# =============================================================================
_HERE = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
LCTSC_DICOM_ROOT = _HERE / "data" / "LCTSC"          # folder that contains LCTSC-Train-* / LCTSC-Test-*
OUTPUT_ROOT = _HERE / "outputs" / "LCTSC_ELBOW3D_OUTPUTS"

DEVELOPMENT_SUBJECTS = ["LCTSC-Test-S3-104"]
MAX_SUBJECTS = None        # None = all 60 subjects (use e.g. 2 for a quick test)

RUN_PROPOSED = True
RUN_CLASSICAL = True
RUN_LUNGMASK = True
LUNGMASK_MODELS = ["R231", "R231CovidWeb", "LTRCLobes"]
RUN_TOTALSEGMENTATOR = True
TOTALSEG_FAST = False
RUN_EVALUATION = True
RUN_STATISTICS = True
N_JOBS = -1
# =============================================================================

PACKAGE_DIR = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
if str(PACKAGE_DIR) not in sys.path:
    sys.path.insert(0, str(PACKAGE_DIR))

import numpy as np                                   # noqa: E402
import pandas as pd                                  # noqa: E402
import SimpleITK as sitk                             # noqa: E402

from lungseg import comparison as C                  # noqa: E402
from lungseg.config import Config                    # noqa: E402
from lungseg.io_utils import load_volume             # noqa: E402
from lungseg.pipeline import run_volume              # noqa: E402

METHODS = ["A5_final", "A5_no_cavity",
           "B_lungmask_R231", "B_lungmask_R231CovidWeb", "B_lungmask_LTRCLobes",
           "B_totalsegmentator", "C_otsu3d", "C_hu2001"]
HU_FLOOR = -1024.0   # LCTSC stores -2000/-3024 outside the scan field of view;
                     # every method receives the same clipped volume.


def frozen_config() -> Config:
    covid = runpy.run_path(str(PACKAGE_DIR / "RUN_IN_SPYDER.py"), run_name="settings_only")
    f = covid["OUTPUT_ROOT"] / "frozen_config.json"
    if not f.exists():
        raise FileNotFoundError(f"{f} not found - the LCTSC run must reuse the frozen COVID config.")
    cfg = Config.from_json(f)
    if cfg.config_version != Config().config_version:
        raise RuntimeError("frozen_config.json is from another code version.")
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    cfg.to_json(OUTPUT_ROOT / "frozen_config.json")
    print(f"Frozen configuration reused from {f}")
    cfg.n_jobs = N_JOBS
    return cfg


def lctsc_helper():
    # DICOM / RTSTRUCT loading functions of the authors' earlier LCTSC package,
    # copied verbatim into lungseg/lctsc_io.py (subject filter off: all 60 subjects).
    from lungseg import lctsc_io as mod
    mod.SUBJECT_FILTER = None
    mod.MAX_SUBJECTS = None
    return mod


def prepare_cases():
    cache = OUTPUT_ROOT / "_hu_input"
    cache.mkdir(parents=True, exist_ok=True)
    listing = cache / "subjects_v2.json"   # v2: written with the subject filter OFF
    known = json.loads(listing.read_text()) if listing.exists() else {}
    core = None
    if not known:                       # DICOM scan only once (cached)
        core = lctsc_helper()
        subjects = core.discover_lctsc_subjects(LCTSC_DICOM_ROOT)
        known = {s.patient_id: {"ct_files": [str(p) for p in s.ct_files],
                                "rtstruct": str(s.rtstruct_path) if s.rtstruct_path else None}
                 for s in subjects}
        listing.write_text(json.dumps(known))
        print(f"  DICOM scan: {len(known)} subjects found in {LCTSC_DICOM_ROOT}")
        if len(known) < 60:
            print("  WARNING: LCTSC has 60 subjects - check that the download is complete.")
    ids = sorted(known)[: MAX_SUBJECTS] if MAX_SUBJECTS else sorted(known)
    cases = []
    for pid in ids:
        hu_path = cache / f"{pid}.nii.gz"
        ref = OUTPUT_ROOT / "_reference" / f"{pid}.nii.gz"
        hu_img, ref_img = _read_or_delete(hu_path), _read_or_delete(ref)
        if hu_img is None or ref_img is None:
            _check_disk(cache)
            core = core or lctsc_helper()
            img = core.read_ct_series([Path(p) for p in known[pid]["ct_files"]])
            if hu_img is None:
                arr = np.maximum(sitk.GetArrayFromImage(img), HU_FLOOR)
                out = sitk.GetImageFromArray(np.rint(arr).astype(np.int16))
                out.CopyInformation(img)
                _atomic(lambda t: sitk.WriteImage(out, str(t), useCompression=True), hu_path)
                hu_img = sitk.ReadImage(str(hu_path))
            if ref_img is None:
                subj = core.SubjectInfo(pid, "", [Path(p) for p in known[pid]["ct_files"]],
                                        Path(known[pid]["rtstruct"]) if known[pid]["rtstruct"] else None)
                left, right, _ = core.load_reference_masks_rtstruct(subj, img, img)
                ref.parent.mkdir(parents=True, exist_ok=True)
                _atomic(lambda t: core.save_mask(left | right, img, t), ref)
                ref_img = sitk.ReadImage(str(ref))
                print(f"  {pid}: reference mask created from RTSTRUCT")
        if hu_img.GetSize() != ref_img.GetSize() or \
                not np.allclose(hu_img.GetOrigin(), ref_img.GetOrigin(), atol=1e-2):
            print(f"  {pid}: CT / reference geometry mismatch - excluded")
            continue
        cases.append({"case": pid, "hu_path": hu_path, "gt": ref, "lesion": None})
    return cases


def _read_or_delete(p: Path):
    """Read a cached NIfTI completely; a truncated/corrupt file (e.g. from a
    full disk) is deleted so that it is rebuilt."""
    if not _exists(p):
        return None
    try:
        with gzip.open(p, "rb") as fh:          # a truncated .gz fails here
            while fh.read(1 << 24):
                pass
        return sitk.ReadImage(str(p))
    except (RuntimeError, EOFError, OSError, zlib.error):
        print(f"  {p.parent.name}/{p.name}: incomplete/corrupt file - deleted and rebuilt")
        try:
            p.unlink()
        except OSError:
            pass
        return None


def _atomic(write, path: Path):
    """Write to a temporary name first, so a crash never leaves a half file."""
    tmp = path.with_name(path.name.replace(".nii.gz", "") + "_tmp.nii.gz")
    write(tmp)
    os.replace(tmp, path)


def _check_disk(folder: Path, need_gb: float = 2.0):
    free = shutil.disk_usage(folder).free / 1e9
    if free < need_gb:
        raise RuntimeError(
            f"Only {free:.1f} GB free on the drive of {folder}. The LCTSC run needs about "
            "6 GB (CT cache ~2 GB + TotalSegmentator temporary files). Free some space "
            "or set OUTPUT_ROOT to a drive with more space, then run again.")


def _exists(p: Path) -> bool:
    try:
        return p.exists()
    except OSError:          # drive not connected
        return False


def statistics(df, rep):
    df = df.copy()
    df["development_case"] = df.case.isin(DEVELOPMENT_SUBJECTS)
    df["site"] = df.case.str.extract(r"(S\d)")[0]
    df.to_csv(rep / "comparison_metrics_all_cases.csv", index=False)
    ext = df[~df.development_case & df.method.isin(METHODS)]
    sets = {"stats_comparison": ext}
    for site in sorted(ext.site.dropna().unique()):
        sets[f"stats_comparison_{site}"] = ext[ext.site == site]
    for name, sub in sets.items():
        if sub.case.nunique() < 5:
            continue
        csv = rep / f"{name}.csv"
        sub.to_csv(csv, index=False)
        argv = sys.argv
        sys.argv = ["03_statistics.py", "--csv", str(csv), "--out", str(rep / name)]
        try:
            print(f"\n=== {name} (n = {sub.case.nunique()} subjects) ===")
            runpy.run_path(str(PACKAGE_DIR / "scripts" / "03_statistics.py"), run_name="__main__")
        finally:
            sys.argv = argv
    C.dice_boxplot(ext, METHODS, rep / "figures" / "dice_boxplot_lctsc.png",
                   f"LCTSC (n = {ext.case.nunique()})")


def main():
    t0 = time.perf_counter()
    rep = OUTPUT_ROOT / "_reports"
    rep.mkdir(parents=True, exist_ok=True)
    cfg = frozen_config()
    print(f"Free disk space for outputs: {shutil.disk_usage(OUTPUT_ROOT).free / 1e9:.1f} GB")
    _check_disk(OUTPUT_ROOT, 3.0)
    cases = prepare_cases()
    print(f"{len(cases)} LCTSC subjects ready")
    failures = []

    if RUN_PROPOSED:
        for i, c in enumerate(cases, 1):
            out_dir = OUTPUT_ROOT / c["case"]
            if (out_dir / "A5_final.nii.gz").exists():
                continue
            try:
                vol, spacing, affine = load_volume(c["hu_path"])
                meta = run_volume(vol, spacing, affine, out_dir, cfg, True, source=str(c["hu_path"]))
                print(f"[proposed {i}/{len(cases)}] {c['case']}: {meta['timing_s']['total_s']:.0f} s", flush=True)
            except Exception as e:
                C._fail(failures, c["case"], "A5_final", e)
    if RUN_CLASSICAL:
        C.run_classical(cases, OUTPUT_ROOT, False, failures)
    if RUN_LUNGMASK:
        try:
            C.run_lungmask(cases, OUTPUT_ROOT, LUNGMASK_MODELS, False, failures)
        except ImportError:
            print("  lungmask is not installed:  %pip install lungmask")
    if RUN_TOTALSEGMENTATOR:
        try:
            C.run_totalsegmentator(cases, OUTPUT_ROOT, TOTALSEG_FAST, False, failures)
        except ImportError:
            print("  TotalSegmentator is not installed:  %pip install TotalSegmentator")
    if failures:
        pd.DataFrame(failures).to_csv(rep / "lctsc_failures.csv", index=False)
        print(f"{len(failures)} failure(s) - see {rep / 'lctsc_failures.csv'}")

    if RUN_EVALUATION:
        print("\n--- evaluation ---")
        df = C.evaluate(cases, OUTPUT_ROOT, METHODS, rep / "comparison_metrics_cache.csv")
        rt = C.runtime_table(OUTPUT_ROOT, cases, METHODS)
        if not rt.empty:
            rt.to_csv(rep / "runtime_cpu_s.csv", index=False)
            print("\nCPU time per case (s):\n" + rt.round(1).to_string(index=False))
        if RUN_STATISTICS and not df.empty:
            statistics(df, rep)
    print(f"\nDONE in {(time.perf_counter() - t0) / 3600:.2f} h. Results: {rep}")


if __name__ == "__main__":
    main()
