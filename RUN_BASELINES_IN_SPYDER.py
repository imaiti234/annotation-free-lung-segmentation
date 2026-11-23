# -*- coding: utf-8 -*-
"""
RUN_BASELINES_IN_SPYDER.py - deep-learning comparators (pretrained, CPU)
=========================================================================

lungmask (Hofmanninger et al., Eur Radiol Exp 2020, 4:50):
  R231          - U-Net trained on 231 diverse routine CT scans
  R231CovidWeb  - R231 fine-tuned with COVID-19 cases

Install once (Spyder console):   %pip install lungmask
Then press F5. Masks are written next to your method's output
(<OUTPUT_ROOT>/<case>/B_lungmask_R231.nii.gz ...). Afterwards run
RUN_IN_SPYDER.py again: segmentation is skipped (already done) and the
baselines are evaluated and compared with the same metrics and tests.

The same intensity decoding as for the proposed method is used, so all methods
see identical HU input. CPU run time per case is saved for the cost table.
"""
from __future__ import annotations

import json
import runpy
import sys
import time
from pathlib import Path

import numpy as np
import SimpleITK as sitk

PACKAGE_DIR = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
settings = runpy.run_path(str(PACKAGE_DIR / "RUN_IN_SPYDER.py"), run_name="settings_only")
CT_DIR, OUTPUT_ROOT = settings["CT_DIR"], settings["OUTPUT_ROOT"]
decode_to_hu, list_nii, nii_stem = settings["decode_to_hu"], settings["list_nii"], settings["nii_stem"]

MODELS = ["R231", "R231CovidWeb"]
OVERWRITE = False


def main():
    from lungmask import LMInferer
    timings = {}
    files = list_nii(CT_DIR)
    print(f"{len(files)} CT volumes")
    for model in MODELS:
        inferer = LMInferer(modelname=model, force_cpu=True)
        for f in files:
            cid = nii_stem(f)
            out = OUTPUT_ROOT / cid / f"B_lungmask_{model}.nii.gz"
            if out.exists() and not OVERWRITE:
                continue
            img = sitk.ReadImage(str(f), sitk.sitkFloat32)
            arr = sitk.GetArrayFromImage(img)
            hu, mode = decode_to_hu(cid, arr)
            hu_img = sitk.GetImageFromArray(hu.astype(np.int16))
            hu_img.CopyInformation(img)
            t = time.perf_counter()
            seg = inferer.apply(hu_img)
            timings[f"{model}/{cid}"] = time.perf_counter() - t
            seg_img = sitk.GetImageFromArray(seg.astype(np.uint8))
            seg_img.CopyInformation(img)
            out.parent.mkdir(parents=True, exist_ok=True)
            sitk.WriteImage(seg_img, str(out), useCompression=True)
            print(f"[{model}] {cid}: {timings[f'{model}/{cid}']:.1f} s ({mode})", flush=True)
    rep = OUTPUT_ROOT / "_reports"
    rep.mkdir(parents=True, exist_ok=True)
    (rep / "baseline_cpu_time_s.json").write_text(json.dumps(timings, indent=2))
    print("Done. Now run RUN_IN_SPYDER.py again to evaluate all methods together.")


if __name__ == "__main__":
    main()
