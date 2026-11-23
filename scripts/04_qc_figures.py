"""
STEP 4 - Qualitative figures for the paper.

    python scripts/04_qc_figures.py --image data/medseg/rp_im/1.nii.gz \
        --pred-dir outputs/medseg/1 --gt-lung data/medseg/rp_lung_msk/1.nii.gz \
        --gt-lesion data/medseg/rp_msk/1.nii.gz --out figures/case1 --n-slices 5

Creates
  qc_comparison.png   Raw | GT (+ lesion outline) | manuscript pipeline | proposed
  stages.png          one slice through all stages (ablation figure)
  elbow_slice<z>.png  area-vs-threshold curve with the selected Th1*
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402
import numpy as np                # noqa: E402
from skimage import measure       # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lungseg.io_utils import load_mask, load_volume, to_display   # noqa: E402

WINDOW = (-1350, 150)
COLORS = {"gt": "#00e5ff", "lesion": "#ff3b30", "A5_no_cavity": "#ff9f0a", "A5_final": "#30d158"}


def outline(ax, mask, color, lw=1.2):
    for c in measure.find_contours(to_display(mask.astype(float)), 0.5):
        ax.plot(c[:, 1], c[:, 0], color=color, lw=lw)


def show(ax, img, title):
    ax.imshow(to_display(img), cmap="gray", vmin=WINDOW[0], vmax=WINDOW[1])
    ax.set_title(title, fontsize=10)
    ax.axis("off")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", type=Path, required=True)
    ap.add_argument("--pred-dir", type=Path, required=True)
    ap.add_argument("--gt-lung", type=Path, default=None)
    ap.add_argument("--gt-lesion", type=Path, default=None)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--slices", type=int, nargs="*", default=None)
    ap.add_argument("--n-slices", type=int, default=5)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    ct, _, _ = load_volume(args.image)
    gt = load_mask(args.gt_lung) > 0 if args.gt_lung else None
    lesion = load_mask(args.gt_lesion) if args.gt_lesion else None
    base = load_mask(args.pred_dir / "A5_no_cavity.nii.gz") > 0
    final = load_mask(args.pred_dir / "A5_final.nii.gz") > 0

    if args.slices:
        zs = args.slices
    elif lesion is not None:
        zs = list(np.argsort((lesion > 0).reshape(lesion.shape[0], -1).sum(1))[::-1][: args.n_slices])
        zs = sorted(int(z) for z in zs)
    elif gt is not None:
        miss = (gt & ~final).reshape(gt.shape[0], -1).sum(1)
        zs = sorted(int(z) for z in np.argsort(miss)[::-1][: args.n_slices])
    else:
        zs = sorted(int(z) for z in np.linspace(0, ct.shape[0] - 1, args.n_slices + 2)[1:-1])

    ncol = 4 if gt is not None else 3
    fig, axes = plt.subplots(len(zs), ncol, figsize=(3.2 * ncol, 3.2 * len(zs)), squeeze=False)
    for r, z in enumerate(zs):
        c = 0
        show(axes[r, c], ct[z], f"Raw CT | slice {z}"); c += 1
        if gt is not None:
            show(axes[r, c], ct[z], "Ground truth")
            outline(axes[r, c], gt[z], COLORS["gt"])
            if lesion is not None:
                outline(axes[r, c], lesion[z] > 0, COLORS["lesion"], lw=0.8)
            c += 1
        show(axes[r, c], ct[z], "Manuscript (A1-A5)")
        outline(axes[r, c], base[z], COLORS["A5_no_cavity"]); c += 1
        show(axes[r, c], ct[z], "Proposed (+A4b)")
        outline(axes[r, c], final[z], COLORS["A5_final"])
    fig.tight_layout()
    fig.savefig(args.out / "qc_comparison.png", dpi=200)
    plt.close(fig)

    # stage-by-stage figure on the most affected slice
    z = zs[len(zs) // 2]
    stage_files = ["A1_coarse", "A2_elbow", "A3_separation", "A4_merging", "A4b_cavity", "A5_final"]
    stage_files = [s for s in stage_files if (args.pred_dir / f"{s}.nii.gz").exists()]
    fig, axes = plt.subplots(1, len(stage_files), figsize=(3.2 * len(stage_files), 3.4))
    for ax, s in zip(np.atleast_1d(axes), stage_files):
        m = load_mask(args.pred_dir / f"{s}.nii.gz")
        show(ax, ct[z], s.replace("_", " "))
        if m.max() >= 2:
            outline(ax, m[z] == 1, "#30d158")
            outline(ax, m[z] == 2, "#0a84ff")
        else:
            outline(ax, m[z] > 0, "#30d158")
    fig.tight_layout()
    fig.savefig(args.out / "stages.png", dpi=200)
    plt.close(fig)

    curves = args.pred_dir / "elbow_curves.npz"
    if curves.exists():
        d = np.load(curves)
        fig, ax = plt.subplots(figsize=(4.2, 3.2))
        ax.plot(d["sweep"], d["areas"][z], "o-", ms=3, color="#444")
        ax.axvline(d["threshold"][z], color="#ff3b30", ls="--", label=f"Th1* = {d['threshold'][z]:.0f}")
        ax.set_xlabel("Threshold (HU)")
        ax.set_ylabel("Lung area (pixels)")
        ax.set_title(f"Elbow curve, slice {z}")
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(args.out / f"elbow_slice{z}.png", dpi=200)
        plt.close(fig)
    print(f"figures written to {args.out}")


if __name__ == "__main__":
    main()
