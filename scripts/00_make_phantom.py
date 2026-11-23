"""
STEP 0 (optional) - Synthetic chest CT phantom for a quick installation test.

Creates a thorax with rib cage, spine, heart, two lungs and a large posterior
(dependent) ground-glass lesion with a consolidation core touching the chest
wall - the failure mode seen on MedSeg. Writes the MedSeg folder layout:
    data/phantom/rp_im/phantom1.nii.gz
    data/phantom/rp_lung_msk/phantom1.nii.gz
    data/phantom/rp_msk/phantom1.nii.gz   (1 = GGO, 2 = consolidation)
"""
from __future__ import annotations

import argparse
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy.ndimage import gaussian_filter


def ellipse(yy, xx, cy, cx, ry, rx):
    return ((yy - cy) / ry) ** 2 + ((xx - cx) / rx) ** 2 <= 1.0


def make(nz=36, n=256, seed=0):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:n, 0:n].astype(float)
    ct = np.full((nz, n, n), -1000.0)
    lung_gt = np.zeros((nz, n, n), np.uint8)
    lesion = np.zeros((nz, n, n), np.uint8)
    cy, cx = 128, 128
    for z in range(nz):
        t = np.sin(np.pi * (z + 1) / (nz + 1))          # lungs taper at apex/base
        s = ct[z]
        s[ellipse(yy, xx, cy, cx, 85, 115)] = 30        # skin
        s[ellipse(yy, xx, cy, cx, 83, 113)] = -100      # subcutaneous fat
        s[ellipse(yy, xx, cy, cx, 76, 106)] = 40        # muscle / chest wall
        s[(yy < 40) & (np.abs(xx - cx) < 120)] = -1000
        s[(yy > 26) & (yy < 31) & (np.abs(xx - cx) < 115)] = 300   # table (posterior)
        for ang in np.linspace(0, 2 * np.pi, 22, endpoint=False):  # ribs
            ry, rx = cy + 70 * np.sin(ang), cx + 99 * np.cos(ang)
            s[ellipse(yy, xx, ry, rx, 4, 6)] = 700
        s[ellipse(yy, xx, cy - 62, cx, 12, 11)] = 700  # vertebra (posterior = low y)
        right = ellipse(yy, xx, cy + 2, cx + 50, 50 * t + 4, 36 * t + 3)
        left = ellipse(yy, xx, cy + 2, cx - 50, 50 * t + 4, 34 * t + 3)
        heart = ellipse(yy, xx, cy + 10, cx - 8, 42, 34)
        right &= ~heart
        left &= ~heart
        lung = right | left
        s[lung] = -860 + rng.normal(0, 25, lung.sum())
        s[heart] = 40
        vessels = lung & (rng.random((n, n)) < 0.004)
        s[vessels] = 30
        lung_gt[z][right] = 1
        lung_gt[z][left] = 2
        if 8 <= z < nz - 8:                                # dependent lesion, both lungs
            for side, cxs in ((right, cx + 50), (left, cx - 50)):
                ggo = side & (yy < cy - 10) & ellipse(yy, xx, cy - 45, cxs, 32, 45)
                cons = ggo & ellipse(yy, xx, cy - 40, cxs + 5, 8, 14)
                s[ggo] = -300 + rng.normal(0, 90, ggo.sum())
                s[cons] = 20 + rng.normal(0, 25, cons.sum())
                lesion[z][ggo] = 1
                lesion[z][cons] = 2
    ct = gaussian_filter(ct, sigma=(0, 1.2, 1.2))   # partial-volume blur
    ct += rng.normal(0, 15, ct.shape)
    return ct.astype(np.int16), lung_gt, lesion


def save(arr_zyx, path, spacing=(0.75, 0.75, 2.5)):
    path.parent.mkdir(parents=True, exist_ok=True)
    aff = np.diag([spacing[0], spacing[1], spacing[2], 1.0])
    nib.save(nib.Nifti1Image(np.ascontiguousarray(arr_zyx.transpose(2, 1, 0)), aff), str(path))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("data/phantom"))
    ap.add_argument("--n", type=int, default=2)
    a = ap.parse_args()
    for i in range(1, a.n + 1):
        ct, lung, les = make(seed=i)
        save(ct, a.out / "rp_im" / f"phantom{i}.nii.gz")
        save(lung, a.out / "rp_lung_msk" / f"phantom{i}.nii.gz")
        save(les, a.out / "rp_msk" / f"phantom{i}.nii.gz")
    print(f"phantoms written to {a.out}")
