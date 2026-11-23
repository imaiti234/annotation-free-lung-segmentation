"""NIfTI input/output. Volumes are handled internally as (Z, Y, X) arrays in RAS+
orientation, i.e. column index increases towards the patient's RIGHT and row index
increases towards ANTERIOR."""
from __future__ import annotations

from pathlib import Path
from typing import Tuple

import nibabel as nib
import numpy as np


def load_volume(path) -> Tuple[np.ndarray, Tuple[float, float, float], np.ndarray]:
    """Return (volume[Z,Y,X] float32, spacing (sz, sy, sx) in mm, canonical affine)."""
    img = nib.as_closest_canonical(nib.load(str(path)))
    data = np.asarray(img.get_fdata(dtype=np.float32))
    if data.ndim == 4:
        data = data[..., 0]
    sx, sy, sz = (float(v) for v in img.header.get_zooms()[:3])
    return np.ascontiguousarray(data.transpose(2, 1, 0)), (sz, sy, sx), img.affine


def load_mask(path) -> np.ndarray:
    vol, _, _ = load_volume(path)
    return np.rint(vol).astype(np.int16)


def save_mask(mask_zyx: np.ndarray, affine: np.ndarray, path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    arr = np.ascontiguousarray(mask_zyx.transpose(2, 1, 0)).astype(np.uint8)
    nib.save(nib.Nifti1Image(arr, affine), str(path))


def to_display(slice_yx: np.ndarray) -> np.ndarray:
    """Radiological display: anterior at top, patient right on viewer's left."""
    return slice_yx[::-1, ::-1]
