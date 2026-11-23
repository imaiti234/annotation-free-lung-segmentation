"""
STEP 2 - Evaluate every saved stage (and any baseline) against the ground truth.

    python scripts/02_evaluate.py --pred outputs/medseg \
        --gt-lung data/medseg/rp_lung_msk --gt-lesion data/medseg/rp_msk \
        --csv outputs/medseg_metrics.csv

Every *.nii.gz inside outputs/medseg/<case>/ is treated as a method/stage, so a
baseline saved as outputs/medseg/<case>/B_lungmask_R231.nii.gz is evaluated too.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lungseg.io_utils import load_mask, load_volume                       # noqa: E402
from lungseg.metrics import affected_slice_recall, lesion_inclusion, overlap_metrics  # noqa: E402


def find(folder: Path, cid: str):
    for ext in (".nii.gz", ".nii"):
        p = folder / f"{cid}{ext}"
        if p.exists():
            return p
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", type=Path, required=True)
    ap.add_argument("--gt-lung", type=Path, required=True)
    ap.add_argument("--gt-lesion", type=Path, default=None)
    ap.add_argument("--csv", type=Path, required=True)
    args = ap.parse_args()

    rows = []
    for case_dir in sorted(p for p in args.pred.iterdir() if p.is_dir()):
        cid = case_dir.name
        gt_path = find(args.gt_lung, cid)
        if gt_path is None:
            print(f"[{cid}] no GT lung mask, skipped")
            continue
        _, spacing, _ = load_volume(gt_path)
        gt = load_mask(gt_path) > 0
        lesion = None
        if args.gt_lesion is not None:
            lp = find(args.gt_lesion, cid)
            lesion = load_mask(lp) if lp else None

        for pred_path in sorted(case_dir.glob("*.nii.gz")):
            method = pred_path.name.replace(".nii.gz", "")
            pred = load_mask(pred_path) > 0
            if pred.shape != gt.shape:
                print(f"[{cid}/{method}] shape mismatch {pred.shape} vs {gt.shape}, skipped")
                continue
            row = {"case": cid, "method": method}
            row.update(overlap_metrics(pred, gt, spacing))
            if lesion is not None:
                row.update(lesion_inclusion(pred, lesion))
                row["recall_affected_slices"] = affected_slice_recall(pred, gt, lesion)
            rows.append(row)
            print(f"[{cid}/{method}] dice={row['dice']:.4f} recall={row['recall']:.4f}"
                  + (f" LIR={row.get('lir_all_parenchymal', np.nan):.3f}" if lesion is not None else ""))

    df = pd.DataFrame(rows)
    args.csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.csv, index=False)
    print(f"wrote {args.csv} ({len(df)} rows)")


if __name__ == "__main__":
    main()
