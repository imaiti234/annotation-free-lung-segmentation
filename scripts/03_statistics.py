"""
STEP 3 - Summary table, ablation and paired statistics for the paper.

    python scripts/03_statistics.py --csv outputs/medseg_metrics.csv --out outputs/stats

Produces:
  summary.csv / summary.tex   mean +/- SD, median [IQR], bootstrap 95 % CI
  paired_tests.csv            A5_final vs every other method/stage,
                              Wilcoxon signed-rank, Holm-corrected
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

METRICS = {  # name: higher_is_better
    "dice": True, "iou": True, "precision": True, "recall": True,
    "hd95_mm": False, "assd_mm": False, "abs_vol_err_pct": False,
    "lir_all_parenchymal": True, "lir_infection": True, "lir_ggo": True, "lir_consolidation": True,
    "recall_affected_slices": True, "dice_affected_slices": True, "dice_lesion_region": True,
}
ORDER = ["A1_coarse", "A2_elbow", "A3_separation", "A4_merging", "A5_no_cavity",
         "A4b_cavity", "A4d_lesion_model", "A5_connectivity", "A5_final"]


def boot_ci(x, n=5000, seed=0):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    if x.size < 2:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    means = rng.choice(x, size=(n, x.size), replace=True).mean(1)
    return np.percentile(means, [2.5, 97.5])


def holm(p):
    p = np.asarray(p, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    m = len(p)
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i])
        adj[i] = min(running, 1.0)
    return adj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--reference", default="A5_final")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.csv)
    metrics = [m for m in METRICS if m in df.columns]
    methods = [m for m in ORDER if m in df.method.unique()] + \
              sorted(set(df.method.unique()) - set(ORDER))

    rows, tex = [], []
    for meth in methods:
        sub = df[df.method == meth]
        r = {"method": meth, "n": len(sub)}
        cells = []
        for m in metrics:
            x = sub[m].astype(float)
            lo, hi = boot_ci(x)
            r.update({f"{m}_mean": x.mean(), f"{m}_sd": x.std(), f"{m}_median": x.median(),
                      f"{m}_q1": x.quantile(.25), f"{m}_q3": x.quantile(.75),
                      f"{m}_ci_lo": lo, f"{m}_ci_hi": hi})
            scale = 100 if m in ("dice", "iou", "precision", "recall") or m.startswith(("lir", "recall_")) else 1
            cells.append(f"{scale * x.mean():.2f} $\\pm$ {scale * x.std():.2f}")
        rows.append(r)
        tex.append(meth.replace("_", "\\_") + " & " + " & ".join(cells) + " \\\\")
    pd.DataFrame(rows).to_csv(args.out / "summary.csv", index=False)
    header = "Method & " + " & ".join(m.replace("_", "\\_") for m in metrics) + " \\\\ \\hline"
    (args.out / "summary.tex").write_text(header + "\n" + "\n".join(tex) + "\n")

    ref = df[df.method == args.reference].set_index("case")
    tests = []
    for meth in methods:
        if meth == args.reference:
            continue
        other = df[df.method == meth].set_index("case")
        common = ref.index.intersection(other.index)
        for m in metrics:
            a, b = ref.loc[common, m].astype(float), other.loc[common, m].astype(float)
            ok = ~(a.isna() | b.isna())
            a, b = a[ok], b[ok]
            if len(a) < 5 or np.allclose(a, b):
                p = np.nan
            else:
                p = wilcoxon(a, b).pvalue
            better = (a > b) if METRICS[m] else (a < b)
            tests.append({"reference": args.reference, "comparator": meth, "metric": m,
                          "n": len(a), "mean_diff": float((a - b).mean()) if len(a) else np.nan,
                          "improved": int(better.sum()), "worse": int(((a != b) & ~better).sum()),
                          "p_wilcoxon": p})
    t = pd.DataFrame(tests)
    if not t.empty:
        valid = t.p_wilcoxon.notna()
        t.loc[valid, "p_holm"] = holm(t.loc[valid, "p_wilcoxon"].values)
    t.to_csv(args.out / "paired_tests.csv", index=False)
    print(pd.DataFrame(rows)[["method", "n"] + [f"{m}_mean" for m in metrics[:6]]].round(4).to_string())
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
