"""
STEP 5 (baseline) - Pretrained deep-learning comparator: lungmask R231
(Hofmanninger et al., Eur Radiol Exp 2020). Runs on CPU, no training needed.

    pip install lungmask SimpleITK
    python scripts/05_baseline_lungmask.py --images data/medseg/rp_im --out outputs/medseg

Saves outputs/medseg/<case>/B_lungmask_R231.nii.gz, which 02_evaluate.py picks
up automatically. Run time is recorded for the CPU-cost comparison table.
Use model "R231CovidWeb" as a second comparator (trained with COVID data).
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import SimpleITK as sitk
from lungmask import LMInferer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--model", default="R231")
    ap.add_argument("--pattern", default="*.nii*")
    args = ap.parse_args()
    inferer = LMInferer(modelname=args.model, force_cpu=True)
    timings = {}
    for f in sorted(args.images.glob(args.pattern)):
        cid = f.name.replace(".nii.gz", "").replace(".nii", "")
        img = sitk.ReadImage(str(f))
        t = time.perf_counter()
        seg = inferer.apply(img)
        timings[cid] = time.perf_counter() - t
        out = sitk.GetImageFromArray(seg.astype("uint8"))
        out.CopyInformation(img)
        (args.out / cid).mkdir(parents=True, exist_ok=True)
        sitk.WriteImage(out, str(args.out / cid / f"B_lungmask_{args.model}.nii.gz"))
        print(f"[{cid}] {timings[cid]:.1f} s")
    (args.out / f"timing_lungmask_{args.model}.json").write_text(json.dumps(timings, indent=2))


if __name__ == "__main__":
    main()
