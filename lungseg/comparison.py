"""Shared code for the method comparison (used by RUN_COMPARISON_IN_SPYDER.py
and RUN_LCTSC_IN_SPYDER.py).

Every comparator receives exactly the same input as the proposed method: one
HU NIfTI per case (`hu_path`). Every prediction is evaluated against the same
reference mask, with the same metric code, on the same list of cases.

Comparators
  B_lungmask_R231          U-Net, 231 routine CTs  (Hofmanninger 2020)
  B_lungmask_R231CovidWeb  R231 + COVID-19 web slices (MedSeg)
  B_lungmask_LTRCLobes     U-Net trained on LTRC (lobes merged to whole lung)
  B_totalsegmentator       nnU-Net, TotalSegmentator v2 (5 lobes merged)
  C_otsu3d, C_hu2001       classical non-learning methods (baselines_classical)
"""
from __future__ import annotations

import json
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

from .baselines_classical import METHODS as CLASSICAL
from .io_utils import load_volume, load_mask, save_mask
from .metrics import overlap_metrics, lesion_inclusion, affected_region_metrics, affected_slice_recall

LUNG_LOBES = ["lung_upper_lobe_left", "lung_lower_lobe_left", "lung_upper_lobe_right",
              "lung_middle_lobe_right", "lung_lower_lobe_right"]

PAPER_NAMES = {
    "A5_final": "Proposed (non-learning)",
    "A5_no_cavity": "Manuscript pipeline (A1-A5)",
    "A1_coarse": "Coarse threshold (A1)",
    "B_lungmask_R231": "lungmask U-Net R231",
    "B_lungmask_R231CovidWeb": "lungmask U-Net R231CovidWeb",
    "B_lungmask_LTRCLobes": "lungmask U-Net LTRCLobes",
    "B_totalsegmentator": "TotalSegmentator (nnU-Net)",
    "C_otsu3d": "Otsu + 3D components",
    "C_hu2001": "Hu et al. 2001 (re-impl.)",
    "P_v4_7_1": "Earlier LCTSC-specific version (V4.7.1)",
}


def _timing_file(out_root: Path) -> Path:
    return out_root / "_reports" / "comparator_cpu_time_s.json"


def _load_timings(out_root: Path) -> dict:
    f = _timing_file(out_root)
    t = json.loads(f.read_text()) if f.exists() else {}
    old = out_root / "_reports" / "baseline_cpu_time_s.json"     # from RUN_BASELINES_IN_SPYDER
    if old.exists():
        for k, v in json.loads(old.read_text()).items():
            model, cid = k.split("/", 1)
            t.setdefault(f"B_lungmask_{model}/{cid}", v)
    return t


def _save_timings(out_root: Path, t: dict) -> None:
    f = _timing_file(out_root)
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps(t, indent=2))


# --------------------------------------------------------------------- runners
def run_lungmask(cases, out_root: Path, models, overwrite=False, failures=None):
    import SimpleITK as sitk
    from lungmask import LMInferer
    t = _load_timings(out_root)
    for model in models:
        name = f"B_lungmask_{model}"
        todo = [c for c in cases if overwrite or not (out_root / c["case"] / f"{name}.nii.gz").exists()]
        if not todo:
            print(f"[{name}] all {len(cases)} cases already done")
            continue
        inferer = LMInferer(modelname=model, force_cpu=True)
        for c in todo:
            try:
                img = sitk.ReadImage(str(c["hu_path"]))
                t0 = time.perf_counter()
                seg = inferer.apply(img)
                t[f"{name}/{c['case']}"] = time.perf_counter() - t0
                out = sitk.GetImageFromArray((seg > 0).astype(np.uint8))
                out.CopyInformation(img)
                (out_root / c["case"]).mkdir(parents=True, exist_ok=True)
                sitk.WriteImage(out, str(out_root / c["case"] / f"{name}.nii.gz"), useCompression=True)
                print(f"[{name}] {c['case']}: {t[name + '/' + c['case']]:.1f} s", flush=True)
            except Exception as e:
                _fail(failures, c["case"], name, e)
            _save_timings(out_root, t)


def run_totalsegmentator(cases, out_root: Path, fast=False, overwrite=False, failures=None):
    import nibabel as nib
    from totalsegmentator.python_api import totalsegmentator
    from totalsegmentator.map_to_binary import class_map
    ids = [k for k, v in class_map["total"].items() if v in LUNG_LOBES]
    name = "B_totalsegmentator"
    t = _load_timings(out_root)
    for c in cases:
        out = out_root / c["case"] / f"{name}.nii.gz"
        if out.exists() and not overwrite:
            continue
        try:
            t0 = time.perf_counter()
            seg = totalsegmentator(str(c["hu_path"]), None, ml=True, roi_subset=LUNG_LOBES,
                                   fast=fast, device="cpu", quiet=True)
            t[f"{name}/{c['case']}"] = time.perf_counter() - t0
            arr = np.isin(np.asarray(seg.dataobj), ids).astype(np.uint8)
            out.parent.mkdir(parents=True, exist_ok=True)
            nib.save(nib.Nifti1Image(arr, seg.affine), str(out))
            print(f"[{name}] {c['case']}: {t[f'{name}/' + c['case']]:.1f} s", flush=True)
        except Exception as e:
            _fail(failures, c["case"], name, e)
        _save_timings(out_root, t)


def run_classical(cases, out_root: Path, overwrite=False, failures=None):
    t = _load_timings(out_root)
    for c in cases:
        todo = [n for n in CLASSICAL
                if overwrite or not (out_root / c["case"] / f"{n}.nii.gz").exists()]
        if not todo:
            continue
        vol, spacing, affine = load_volume(c["hu_path"])
        for n in todo:
            try:
                t0 = time.perf_counter()
                m = CLASSICAL[n](vol, spacing)
                t[f"{n}/{c['case']}"] = time.perf_counter() - t0
                save_mask(m, affine, out_root / c["case"] / f"{n}.nii.gz")
                print(f"[{n}] {c['case']}: {t[f'{n}/' + c['case']]:.1f} s", flush=True)
            except Exception as e:
                _fail(failures, c["case"], n, e)
        _save_timings(out_root, t)


def _fail(failures, cid, method, e):
    msg = f"{type(e).__name__}: {e}"
    print(f"  FAILED {method} on {cid}: {msg}")
    if failures is not None:
        failures.append({"case": cid, "method": method, "error": msg,
                         "traceback": traceback.format_exc()})


# ------------------------------------------------------------------ evaluation
def _lesion_labels(path):
    if path is None:
        return None
    les = load_mask(path)
    if les.max() >= 3:            # Lung_and_Infection_Mask: 3 = infection
        return (les == 3).astype(np.int16)
    return (les > 0).astype(np.int16)


def evaluate(cases, out_root: Path, methods, csv_path: Path, extra_pred=None):
    """Evaluate every available prediction. Results are cached in csv_path and
    recomputed only when a prediction file changed or the lesion mask became
    available. `extra_pred(case, method)` may return a path outside out_root."""
    cache = pd.read_csv(csv_path) if csv_path.exists() else pd.DataFrame()
    keep, rows = [], []
    for i, c in enumerate(cases, 1):
        gt = les = spacing = None
        for meth in methods:
            p = extra_pred(c, meth) if extra_pred else None
            p = p or (out_root / c["case"] / f"{meth}.nii.gz")
            if not Path(p).exists():
                continue
            mtime = Path(p).stat().st_mtime
            has_les = c.get("lesion") is not None
            if not cache.empty:
                hit = cache[(cache.case == c["case"]) & (cache.method == meth)]
                if len(hit) and abs(hit.pred_mtime.iloc[0] - mtime) < 1 and bool(hit.has_lesion.iloc[0]) == has_les:
                    keep.append(hit.iloc[0].to_dict())
                    continue
            if gt is None:
                _, spacing, _ = load_volume(c["gt"])
                gt = load_mask(c["gt"]) > 0
                les = _lesion_labels(c.get("lesion"))
                if les is not None and les.shape != gt.shape:
                    print(f"  {c['case']}: lesion mask shape {les.shape} != GT {gt.shape} - ignored")
                    les = None
            pred = load_mask(p) > 0
            if pred.shape != gt.shape:
                print(f"  {c['case']} {meth}: shape {pred.shape} != GT {gt.shape} - skipped")
                continue
            r = {"case": c["case"], "method": meth, "pred_mtime": mtime, "has_lesion": has_les}
            r.update(overlap_metrics(pred, gt, spacing))
            if les is not None:
                r.update(lesion_inclusion(pred, les, {1: "infection"}))
                r.update(affected_region_metrics(pred, gt, les, spacing))
                r["recall_affected_slices"] = affected_slice_recall(pred, gt, les)
            rows.append(r)
        print(f"  evaluated {i}/{len(cases)}: {c['case']}", flush=True)
    df = pd.DataFrame(keep + rows)
    if not df.empty:
        df.to_csv(csv_path, index=False)
    return df


def runtime_table(out_root: Path, cases, methods) -> pd.DataFrame:
    t = _load_timings(out_root)
    rows = []
    for c in cases:
        meta = out_root / c["case"] / "run_meta.json"
        if meta.exists():
            m = json.loads(meta.read_text())
            tot = m.get("timing_s", {}).get("total_s")
            if tot is not None:
                rows.append({"case": c["case"], "method": "A5_final", "cpu_s": tot})
        for k, v in t.items():
            meth, cid = k.split("/", 1)
            if cid == c["case"] and meth in methods:
                rows.append({"case": cid, "method": meth, "cpu_s": v})
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    s = df.groupby("method").cpu_s.agg(["count", "mean", "std", "median", "min", "max"]).reset_index()
    s.insert(1, "name", s.method.map(PAPER_NAMES).fillna(s.method))
    return s


def dice_boxplot(df: pd.DataFrame, methods, path: Path, title: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    meths = [m for m in methods if m in df.method.unique()]
    if not meths:
        return
    data = [df[df.method == m].dice.values * 100 for m in meths]
    fig, ax = plt.subplots(figsize=(7.0, 0.45 * len(meths) + 1.2))
    bp = ax.boxplot(data, vert=False, widths=0.55, patch_artist=True, showfliers=False,
                    medianprops=dict(color="#1f1f1f", lw=1.5))
    for m, box in zip(meths, bp["boxes"]):
        box.set_facecolor("#2a78d6" if m == "A5_final" else "#d9d9d9")
        box.set_edgecolor("#5f5f5f")
    rng = np.random.default_rng(0)
    for i, d in enumerate(data, 1):
        ax.scatter(d, i + rng.uniform(-0.12, 0.12, d.size), s=12, color="#3d3d3d", zorder=3, lw=0)
    ax.set_yticks(range(1, len(meths) + 1))
    ax.set_yticklabels([PAPER_NAMES.get(m, m) for m in meths])
    ax.invert_yaxis()                       # first method (proposed) on top
    ax.set_xlabel("Dice (%)")
    ax.set_title(title, fontsize=10)
    ax.grid(axis="x", color="#e6e6e6", lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300)
    plt.close(fig)
