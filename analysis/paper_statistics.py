"""
Reproduce the main statistics of the paper from the per-scan results in results/.

    python analysis/paper_statistics.py

Writes analysis/paper_statistics.txt. Needs numpy, pandas and scipy only.

Conventions (as in the paper)
-----------------------------
* COVID-19-CT-Seg: the four development scans are excluded (16 test scans).
  LCTSC: LCTSC-Test-S3-104 (used to develop the original elbow method) is
  excluded (59 test scans). The column `development_case` marks them.
* Paired differences are "proposed minus comparator" in Dice percentage points.
* Bootstrap: 10,000 resamples of scans, percentile intervals. The two-sided 95%
  CI uses the 2.5th and 97.5th percentiles; the one-sided 95% lower bound used
  for the non-inferiority margin of -1 point is the 5th percentile.
* Wilcoxon signed-rank tests (two-sided), Holm-adjusted within each dataset
  (6 comparators on COVID-19-CT-Seg; 7 on LCTSC, including the configuration
  without A4b and A4d).
Bootstrap end points can differ from the paper by about 0.01-0.05 points
because of the random seed.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
OUT = Path(__file__).resolve().parent / "paper_statistics.txt"
B = 10_000
NAMES = {
    "A5_final": "Proposed", "A5_no_cavity": "Without A4b and A4d",
    "C_hu2001": "Hu et al. (re-impl.)", "C_otsu3d": "Otsu + 3D components",
    "B_lungmask_R231": "U-Net R231", "B_lungmask_R231CovidWeb": "U-Net R231CovidWeb",
    "B_lungmask_LTRCLobes": "U-Net LTRCLobes", "B_totalsegmentator": "TotalSegmentator",
}
lines: list[str] = []


def out(s: str = "") -> None:
    print(s)
    lines.append(s)


def holm(p):
    p = np.asarray(p, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for k, i in enumerate(order):
        running = max(running, min(1.0, (len(p) - k) * p[i]))
        adj[i] = running
    return adj


def boot(diff, rng):
    m = rng.choice(diff, (B, diff.size)).mean(1)
    return np.percentile(m, 2.5), np.percentile(m, 5), np.percentile(m, 97.5)


def compare(df: pd.DataFrame, comparators, title: str, rng) -> None:
    dice = df.pivot(index="case", columns="method", values="dice") * 100
    out(f"\n=== {title}: {len(dice)} test scans ===")
    out(f"{'method':24s} {'Dice mean':>9s} {'s.d.':>5s} {'min':>6s} {'n<90':>5s}  {'95% CI':>15s}")
    for m in ["A5_final"] + comparators:
        s = dice[m]
        bm = rng.choice(s.values, (B, s.size)).mean(1)
        lo, hi = np.percentile(bm, [2.5, 97.5])
        out(f"{NAMES[m]:24s} {s.mean():9.2f} {s.std():5.2f} {s.min():6.2f} {int((s < 90).sum()):5d}"
            f"  [{lo:.2f}, {hi:.2f}]")
    rows = []
    for m in comparators:
        d = (dice["A5_final"] - dice[m]).values
        lo2, lo1, hi2 = boot(d, rng)
        rows.append((m, d.mean(), lo1, lo2, hi2, wilcoxon(dice["A5_final"], dice[m]).pvalue,
                     int((d > 0).sum())))
    adj = holm([r[5] for r in rows])
    out(f"\n{'comparator':24s} {'diff':>6s} {'1-sided LB':>10s} {'two-sided 95% CI':>18s}"
        f" {'p':>7s} {'p Holm':>7s} {'prop>comp':>9s}")
    for (m, md, lo1, lo2, hi2, p, npos), a in zip(rows, adj):
        flag = "non-inferior" if lo1 > -1 else "not shown"
        out(f"{NAMES[m]:24s} {md:+6.2f} {lo1:+10.2f}   [{lo2:+.2f}, {hi2:+.2f}] {p:7.4f} {a:7.4f}"
            f" {npos:4d}/{len(d)}   ({flag} at -1 point)")
    g = df.groupby("method")
    sec = pd.DataFrame({
        "recall": g.recall.mean() * 100, "precision": g.precision.mean() * 100,
        "HD95 mean": g.hd95_mm.mean(), "HD95 median": g.hd95_mm.median(),
        "ASSD": g.assd_mm.mean(), "|dV| %": g.abs_vol_err_pct.mean(),
    }).rename(index=NAMES)
    out("\n" + sec.round(2).to_string())


def main() -> None:
    rng = np.random.default_rng(0)
    covid_all = pd.read_csv(RES / "covid19_ct_seg" / "comparison_metrics_all_cases.csv")
    covid = covid_all[~covid_all.development_case]
    nets = ["B_lungmask_R231", "B_lungmask_R231CovidWeb", "B_lungmask_LTRCLobes", "B_totalsegmentator"]
    classical = ["C_hu2001", "C_otsu3d"]
    compare(covid, classical + nets, "COVID-19-CT-Seg (Table 1)", rng)

    # lesion inclusion (fraction of annotated infection inside the predicted lung)
    les = covid.pivot(index="case", columns="method", values="lir_infection") * 100
    out("\nLesion inclusion (%), mean over scans: "
        + ", ".join(f"{NAMES[m]} {les[m].mean():.1f}" for m in ["A5_final", "A5_no_cavity"] + classical + nets))

    # burden split (Table 2) - burden = infected fraction of the reference lung
    sev = pd.read_csv(RES / "covid19_ct_seg" / "severity" / "severity_scan_level.csv")
    sev = sev[~sev.development_case]
    bur = sev.drop_duplicates("case").set_index("case").burden
    med = bur.median()
    dice = covid.pivot(index="case", columns="method", values="dice") * 100
    upper = bur.index[bur > med]
    lower = bur.index[bur <= med]
    out(f"\n=== Burden split at the median ({100 * med:.1f}%) - Table 2 ===")
    for m in ["A5_final", "A5_no_cavity"] + classical + nets:
        rho, p = spearmanr(bur.loc[dice.index], dice[m])
        out(f"{NAMES[m]:24s} rho {rho:+.2f} (p {p:.3f})  lower {dice.loc[lower, m].mean():.2f}"
            f"  upper {dice.loc[upper, m].mean():.2f}")
    for half, idx in (("lower", lower), ("upper", upper)):
        for m in classical + nets + ["A5_no_cavity"]:
            d = (dice.loc[idx, "A5_final"] - dice.loc[idx, m]).values
            lo2, lo1, hi2 = boot(d, rng)
            out(f"  {half:5s} half, proposed - {NAMES[m]:22s} {d.mean():+.2f}"
                f"  two-sided [{lo2:+.2f}, {hi2:+.2f}]  one-sided LB {lo1:+.2f}")

    # stage-wise output (Table 4)
    st = pd.read_csv(RES / "covid19_ct_seg" / "metrics_all_stages.csv")
    st = st[~st.development_case & st.method.str.startswith("A")]
    g = st.groupby("method")
    out("\n=== Output after each operator, COVID-19-CT-Seg (Table 4) ===")
    out((pd.DataFrame({"Dice": g.dice.mean() * 100, "recall": g.recall.mean() * 100,
                       "precision": g.precision.mean() * 100, "HD95": g.hd95_mm.mean()})
         .round(2)).to_string())

    lctsc_all = pd.read_csv(RES / "lctsc" / "comparison_metrics_all_cases.csv")
    lctsc = lctsc_all[~lctsc_all.development_case]
    compare(lctsc, ["A5_no_cavity"] + classical + nets, "LCTSC (Table 3)", rng)
    h = lctsc[lctsc.method == "A5_final"].set_index("case").hd95_mm
    out(f"Proposed: {int((h > 30).sum())} scans with HD95 > 30 mm: {', '.join(h.index[h > 30])}")

    out("\n=== Run time per scan, seconds (all scans incl. development; Table 1, Results) ===")
    for ds in ("covid19_ct_seg", "lctsc"):
        rt = pd.read_csv(RES / ds / "runtime_cpu_s.csv")
        rt["name"] = rt.method.map(NAMES).fillna(rt.method)
        out(f"[{ds}]\n" + rt[["name", "count", "median", "mean", "std", "min", "max"]]
            .round(1).to_string(index=False))

    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nSaved to {OUT}")


if __name__ == "__main__":
    main()
