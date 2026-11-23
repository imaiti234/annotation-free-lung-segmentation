# -*- coding: utf-8 -*-
"""
RUN_SEVERITY_IN_SPYDER.py - how do the methods behave when the disease is severe?
=================================================================================

Run AFTER RUN_COMPARISON_IN_SPYDER.py (it needs the predictions of every
method, the reference lung masks and the infection masks). Nothing is
re-computed by the segmentation methods; this script only analyses their
outputs, so it takes a few minutes. Press F5.

Everything below is defined BEFORE looking at the results (no cherry-picking):

 SEVERITY OF A SCAN   infection burden = infected voxels / reference lung voxels
                      (infection masks are used for evaluation only).
 SEVERITY OF A SLICE  local burden = infected pixels / reference lung pixels
                      on that axial slice.

 Analysis 1 (scan level)   Dice, recall, lesion inclusion versus burden for every
                           method; Spearman correlation and the least-squares
                           slope; high-burden half versus low-burden half.
 Analysis 2 (slice level)  every axial slice of the 16 external scans is binned by
                           local burden; Dice and recall per bin and method.
                           This gives thousands of slices instead of 16 scans.
                           95% CIs are cluster bootstrap over scans.
 Analysis 3 (images)       error maps of the most severe scans (missed lung red,
                           false positive yellow).

Outputs (folder <OUTPUT_ROOT>\\_reports\\severity):
    severity_scan_level.csv, severity_slice_level.csv, severity_summary.csv,
    severity_slice_bins.csv, fig_severity_scatter.png / .pdf,
    fig_severity_slicebins.png / .pdf, fig_severity_montage.png / .pdf
"""
from __future__ import annotations

import runpy
import sys
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
if str(PACKAGE_DIR) not in sys.path:
    sys.path.insert(0, str(PACKAGE_DIR))

import numpy as np                                   # noqa: E402
import pandas as pd                                  # noqa: E402
import matplotlib                                    # noqa: E402
import matplotlib.pyplot as plt                      # noqa: E402
from scipy import stats                              # noqa: E402

from lungseg.io_utils import load_volume, load_mask  # noqa: E402
from lungseg import comparison as C                  # noqa: E402

# ============================= SETTINGS =======================================
METHODS = ["A5_final", "A5_no_cavity", "B_lungmask_R231", "B_lungmask_R231CovidWeb",
           "B_totalsegmentator", "B_lungmask_LTRCLobes", "C_hu2001", "C_otsu3d"]
NAMES = {"A5_final": "Proposed", "A5_no_cavity": "Previous pipeline",
         "B_lungmask_R231": "U-Net R231", "B_lungmask_R231CovidWeb": "U-Net R231CovidWeb",
         "B_totalsegmentator": "TotalSegmentator", "B_lungmask_LTRCLobes": "U-Net LTRCLobes",
         "C_hu2001": "Hu et al.", "C_otsu3d": "Otsu + 3D"}
COLORS = {"A5_final": "#1f6fd6", "A5_no_cavity": "#8fb8ee", "B_lungmask_R231": "#d1495b",
          "B_lungmask_R231CovidWeb": "#e8943a", "B_totalsegmentator": "#2a9d8f",
          "B_lungmask_LTRCLobes": "#8d6cab", "C_hu2001": "#777777", "C_otsu3d": "#aaaaaa"}
SLICE_BINS = [0, 0.001, 0.05, 0.20, 0.40, 1.01]            # local burden (fraction)
SLICE_LABELS = ["none", "<5%", "5-20%", "20-40%", ">40%"]
MIN_LUNG_PX = 500                                           # slices with less lung are ignored
N_MONTAGE = 4                                               # most severe scans shown
MONTAGE_METHODS = ["A5_no_cavity", "A5_final", "B_lungmask_R231", "B_totalsegmentator"]
N_BOOT = 10000
# ==============================================================================

S = runpy.run_path(str(PACKAGE_DIR / "RUN_IN_SPYDER.py"), run_name="settings_only")
OUTPUT_ROOT = S["OUTPUT_ROOT"]
DEV = set(S["DEVELOPMENT_CASES"])
REP = OUTPUT_ROOT / "_reports"
OUT = REP / "severity"
OUT.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(0)


def get_cases():
    g = S["build_cases"].__globals__
    g["ONLY_CASES"], g["MAX_CASES"] = None, None
    cases = [c for c in S["build_cases"]() if c["case"] not in DEV]
    hu_dir = OUTPUT_ROOT / "_hu_input"
    for c in cases:
        f = hu_dir / f"{c['case']}.nii.gz"
        c["hu_path"] = f if f.exists() else c["ct"]
        if c["lesion"] is None:
            raise SystemExit("Infection masks not found. See the header of RUN_COMPARISON_IN_SPYDER.py.")
    return cases


def pred_path(case, m):
    return OUTPUT_ROOT / case["case"] / f"{m}.nii.gz"


def slice_stats(pred, gt, les):
    """Per-slice lung Dice, recall and local burden (arrays over z)."""
    p, g, l = pred.astype(bool), gt.astype(bool), les.astype(bool) & gt.astype(bool)
    tp = (p & g).sum((1, 2)); fp = (p & ~g).sum((1, 2)); fn = (~p & g).sum((1, 2))
    ng = g.sum((1, 2)); nl = l.sum((1, 2))
    dice = np.where(2 * tp + fp + fn > 0, 2 * tp / np.maximum(2 * tp + fp + fn, 1), np.nan)
    rec = np.where(ng > 0, tp / np.maximum(ng, 1), np.nan)
    return dice, rec, nl / np.maximum(ng, 1), ng


def cluster_ci(df, col, by="case"):
    """Mean and 95% cluster-bootstrap CI over scans."""
    groups = [x[col].values for _, x in df.groupby(by)]
    allv = np.concatenate(groups) if groups else np.array([])
    if allv.size == 0:
        return np.nan, np.nan, np.nan
    sums = np.array([v.sum() for v in groups]); ns = np.array([v.size for v in groups])
    idx = rng.integers(0, len(groups), (N_BOOT, len(groups)))
    boot = sums[idx].sum(1) / np.maximum(ns[idx].sum(1), 1)
    return allv.mean(), *np.percentile(boot, [2.5, 97.5])


def main():
    cases = get_cases()
    base = pd.read_csv(REP / "comparison_metrics_all_cases.csv")
    base = base[~base.development_case & base.method.isin(METHODS)]
    scan_rows, slice_rows, cache = [], [], {}
    for i, c in enumerate(cases, 1):
        gt = load_mask(c["gt"]) > 0
        les = C._lesion_labels(c["lesion"]) > 0
        hu, _, _ = load_volume(c["hu_path"])
        lung_vox = gt.sum()
        burden = (les & gt).sum() / max(lung_vox, 1)
        dense = ((hu > -500) & gt).sum() / max(lung_vox, 1)       # not-aerated fraction
        scan_rows.append({"case": c["case"], "burden": burden, "nonaerated_frac": dense})
        for m in METHODS:
            p = pred_path(c, m)
            if not p.exists():
                continue
            pr = load_mask(p) > 0
            if pr.shape != gt.shape:
                continue
            d, r, lb, ng = slice_stats(pr, gt, les)
            ok = ng >= MIN_LUNG_PX
            for z in np.where(ok)[0]:
                slice_rows.append({"case": c["case"], "method": m, "z": z, "local_burden": lb[z],
                                   "dice": d[z], "recall": r[z]})
        print(f"  analysed {i}/{len(cases)}: {c['case']}  burden {100*burden:.1f}%", flush=True)

    scan = pd.DataFrame(scan_rows)
    sl = pd.DataFrame(slice_rows)
    sl["bin"] = pd.cut(sl.local_burden, SLICE_BINS, labels=SLICE_LABELS, right=False)
    sl.to_csv(OUT / "severity_slice_level.csv", index=False)

    # ---------------- analysis 1: scan level ----------------
    m_scan = base.merge(scan, on="case")
    m_scan["lesion_inclusion"] = m_scan["lir_infection"]
    m_scan.to_csv(OUT / "severity_scan_level.csv", index=False)
    med = scan.burden.median()
    m_scan["group"] = np.where(m_scan.burden >= med, "high burden", "low burden")
    rows = []
    for m in METHODS:
        x = m_scan[m_scan.method == m]
        if x.empty:
            continue
        rho, prho = stats.spearmanr(x.burden, x.dice)
        slope = np.polyfit(100 * x.burden, 100 * x.dice, 1)[0]      # Dice points per +1% burden
        r = {"method": m, "name": NAMES[m], "n": len(x), "spearman_rho": rho, "p": prho,
             "slope_dice_pts_per_1pct_burden": slope}
        for g in ["low burden", "high burden"]:
            y = x[x.group == g]
            r[f"dice_{g}"] = 100 * y.dice.mean()
            r[f"min_dice_{g}"] = 100 * y.dice.min()
            r[f"recall_{g}"] = 100 * y.recall.mean()
            r[f"lesion_incl_{g}"] = 100 * y.lesion_inclusion.mean()
            r[f"hd95_{g}"] = y.hd95_mm.mean()
        r["drop_dice_high_vs_low"] = r["dice_high burden"] - r["dice_low burden"]
        rows.append(r)
    summ = pd.DataFrame(rows)
    summ.to_csv(OUT / "severity_summary.csv", index=False)
    print(f"\nScan-level summary (median burden = {100*med:.1f}%):")
    print(summ[["name", "spearman_rho", "p", "slope_dice_pts_per_1pct_burden",
                "dice_low burden", "dice_high burden", "min_dice_high burden",
                "lesion_incl_high burden"]].round(3).to_string(index=False))

    # paired: proposed vs others within the high-burden half
    hi = m_scan[m_scan.group == "high burden"].pivot(index="case", columns="method", values="dice")
    pr_rows = []
    for m in METHODS[1:]:
        if m in hi and "A5_final" in hi:
            dlt = 100 * (hi["A5_final"] - hi[m]).dropna().values
            if dlt.size >= 3:
                bm = rng.choice(dlt, (N_BOOT, dlt.size)).mean(1)
                pr_rows.append({"comparator": NAMES[m], "n": dlt.size, "mean_diff_points": dlt.mean(),
                                "ci_lo": np.percentile(bm, 2.5), "ci_hi": np.percentile(bm, 97.5),
                                "proposed_better_in": int((dlt > 0).sum())})
    pd.DataFrame(pr_rows).to_csv(OUT / "severity_paired_high_burden.csv", index=False)
    print("\nPaired difference in the high-burden half (proposed minus comparator):")
    print(pd.DataFrame(pr_rows).round(3).to_string(index=False))

    # ---------------- analysis 2: slice-level bins ----------------
    brow = []
    for m in METHODS:
        for b in SLICE_LABELS:
            x = sl[(sl.method == m) & (sl.bin == b)]
            if x.empty:
                continue
            md, dlo, dhi = cluster_ci(x.dropna(subset=["dice"]), "dice")
            mr, rlo, rhi = cluster_ci(x.dropna(subset=["recall"]), "recall")
            brow.append({"method": m, "name": NAMES[m], "bin": b, "n_slices": len(x),
                         "n_scans": x.case.nunique(), "dice": 100 * md, "dice_lo": 100 * dlo,
                         "dice_hi": 100 * dhi, "recall": 100 * mr, "recall_lo": 100 * rlo,
                         "recall_hi": 100 * rhi})
    bins = pd.DataFrame(brow)
    bins.to_csv(OUT / "severity_slice_bins.csv", index=False)
    print("\nSlice-level Dice (%) by local burden:")
    print(bins.pivot(index="name", columns="bin", values="dice")[SLICE_LABELS].round(2).to_string())
    print("\nSlices per bin:", bins[bins.method == "A5_final"].set_index("bin").n_slices.to_dict())

    # ---------------- figures ----------------
    plt.rcParams.update({"font.size": 8, "axes.linewidth": 0.6})
    # (a) scatter
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 3.0))
    for m in METHODS[:6]:
        x = m_scan[m_scan.method == m]
        for k, (col, lab) in enumerate([("dice", "Dice (%)"), ("lesion_inclusion", "Lesion inclusion (%)")]):
            xs, ys = 100 * x.burden, 100 * x[col]
            ax[k].scatter(xs, ys, s=12, color=COLORS[m], label=NAMES[m], zorder=3,
                          marker="o" if m == "A5_final" else "s", lw=0)
            if len(x) > 3:
                z = np.polyfit(xs, ys, 1); t = np.linspace(xs.min(), xs.max(), 20)
                ax[k].plot(t, np.polyval(z, t), color=COLORS[m], lw=1.2 if m == "A5_final" else 0.8)
            ax[k].set_xlabel("Infection burden (% of lung)"); ax[k].set_ylabel(lab)
            ax[k].grid(color="#eee", lw=0.5); ax[k].set_axisbelow(True)
    ax[0].set_title("a", loc="left", fontweight="bold"); ax[1].set_title("b", loc="left", fontweight="bold")
    ax[0].legend(frameon=False, fontsize=6.5, loc="lower left")
    fig.tight_layout()
    for e in ("png", "pdf"):
        fig.savefig(OUT / f"fig_severity_scatter.{e}", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # (b) slice bins
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 3.0))
    sel = ["A5_final", "A5_no_cavity", "B_lungmask_R231", "B_totalsegmentator", "C_hu2001"]
    w = 0.8 / len(sel)
    for k, (col, lab) in enumerate([("dice", "Slice Dice (%)"), ("recall", "Slice recall (%)")]):
        for j, m in enumerate(sel):
            x = bins[bins.method == m].set_index("bin").reindex(SLICE_LABELS)
            pos = np.arange(len(SLICE_LABELS)) + (j - len(sel) / 2 + 0.5) * w
            err = [x[col] - x[col + "_lo"], x[col + "_hi"] - x[col]]
            ax[k].bar(pos, x[col], w, color=COLORS[m], label=NAMES[m], yerr=err,
                      error_kw=dict(lw=0.6, capsize=1))
        ax[k].set_xticks(range(len(SLICE_LABELS)))
        n = bins[bins.method == "A5_final"].set_index("bin").n_slices.reindex(SLICE_LABELS)
        ax[k].set_xticklabels([f"{b}\n(n={int(v)})" for b, v in zip(SLICE_LABELS, n)])
        ax[k].set_xlabel("Local infection burden of the slice"); ax[k].set_ylabel(lab)
        lo = np.nanmin(bins[col + "_lo"]); ax[k].set_ylim(max(60, lo - 3), 100)
        ax[k].grid(axis="y", color="#eee", lw=0.5); ax[k].set_axisbelow(True)
    ax[0].set_title("a", loc="left", fontweight="bold"); ax[1].set_title("b", loc="left", fontweight="bold")
    ax[0].legend(frameon=False, fontsize=6.5, loc="lower left")
    fig.tight_layout()
    for e in ("png", "pdf"):
        fig.savefig(OUT / f"fig_severity_slicebins.{e}", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # (c) montage of the most severe scans
    worst = scan.sort_values("burden", ascending=False).head(N_MONTAGE).case.tolist()
    ncol = 1 + len(MONTAGE_METHODS)
    fig, axs = plt.subplots(len(worst), ncol, figsize=(1.6 * ncol + 0.4, 1.7 * len(worst)))
    axs = np.atleast_2d(axs)
    for r, cid in enumerate(worst):
        c = next(x for x in cases if x["case"] == cid)
        hu, _, _ = load_volume(c["hu_path"]); gt = load_mask(c["gt"]) > 0
        les = C._lesion_labels(c["lesion"]) > 0
        z = int(np.argmax((les & gt).sum((1, 2))))
        ys, xs = np.where(gt[z] | (gt[max(z - 3, 0):z + 4].any(0)))
        y0, y1, x0, x1 = max(ys.min() - 8, 0), ys.max() + 8, max(xs.min() - 8, 0), xs.max() + 8
        img = np.clip((hu[z, y0:y1, x0:x1] + 1350) / 1500, 0, 1)
        g2 = gt[z, y0:y1, x0:x1]; l2 = les[z, y0:y1, x0:x1]
        rgb = np.dstack([img] * 3)
        rgb[l2 & g2] = 0.65 * rgb[l2 & g2] + 0.35 * np.array([1.0, 0.6, 0.0])
        axs[r, 0].imshow(rgb, cmap="gray"); axs[r, 0].contour(g2, [0.5], colors="cyan", linewidths=0.6)
        axs[r, 0].set_ylabel(f"{cid.replace('_', ' ')}\nburden {100 * float(scan.set_index('case').burden[cid]):.0f}%",
                             fontsize=6)
        for k, m in enumerate(MONTAGE_METHODS, 1):
            p = pred_path(c, m)
            ax = axs[r, k]
            if not p.exists():
                ax.axis("off"); continue
            pr = (load_mask(p) > 0)[z, y0:y1, x0:x1]
            e = np.dstack([img] * 3)
            e[g2 & ~pr] = [1, 0, 0]; e[pr & ~g2] = [1, 0.9, 0]
            ax.imshow(e); ax.contour(pr, [0.5], colors="lime", linewidths=0.5)
            dz = 2 * (pr & g2).sum() / max(pr.sum() + g2.sum(), 1)
            ax.text(0.03, 0.04, f"{dz:.3f}", transform=ax.transAxes, color="w", fontsize=6,
                    bbox=dict(fc="k", alpha=0.5, pad=0.8, lw=0))
            if r == 0:
                ax.set_title(NAMES[m], fontsize=7)
        if r == 0:
            axs[r, 0].set_title("CT + reference\n(infection orange)", fontsize=7)
    for a in axs.ravel():
        a.set_xticks([]); a.set_yticks([])
        for s in a.spines.values():
            s.set_visible(False)
    fig.tight_layout(pad=0.3)
    for e in ("png", "pdf"):
        fig.savefig(OUT / f"fig_severity_montage.{e}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"\nDONE. Results in {OUT}")


if __name__ == "__main__":
    main()
