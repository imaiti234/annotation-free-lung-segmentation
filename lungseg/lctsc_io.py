"""
LCTSC data access: DICOM discovery, CT series reading and RTSTRUCT rasterisation.

These functions are copied verbatim from the authors' earlier LCTSC evaluation
package (file lctsc_adaptive_elbow_anatomical_v4_3_shape_constrained_S3_104.py),
which RUN_LCTSC_IN_SPYDER.py used to load the data for the paper. Only the data-
loading part is kept; none of it is used by the segmentation method itself.
Reference masks are used for evaluation only.

Needs: pydicom, SimpleITK and rt-utils (pip install pydicom SimpleITK rt-utils).
"""
from __future__ import annotations

import logging
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import pydicom
import SimpleITK as sitk

try:
    from rt_utils import RTStructBuilder
except ImportError:
    RTStructBuilder = None

# Settings of the original helper. The paper run switched the subject filter off.
MIN_CT_SLICES = 20
NIFTI_COMPRESSION = True
SUBJECT_FILTER: Optional[str] = None
MAX_SUBJECTS: Optional[int] = None


@dataclass
class SubjectInfo:
    patient_id: str
    ct_series_uid: str
    ct_files: List[Path]
    rtstruct_path: Optional[Path]


def safe_float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def normalize_roi_name(name: str) -> str:
    return "".join(ch.lower() for ch in name if ch.isalnum())


def get_rt_referenced_series_uids(ds: pydicom.dataset.Dataset) -> Set[str]:
    refs: Set[str] = set()
    try:
        for frame in ds.ReferencedFrameOfReferenceSequence:
            for study in frame.RTReferencedStudySequence:
                for series in study.RTReferencedSeriesSequence:
                    uid = getattr(series, "SeriesInstanceUID", None)
                    if uid:
                        refs.add(str(uid))
    except Exception:
        pass
    return refs


def ct_sort_key(ds: pydicom.dataset.Dataset, fallback_path: Path) -> Tuple[float, float, str]:
    """Sort by physical position along the slice normal, then InstanceNumber."""
    try:
        orientation = np.asarray(ds.ImageOrientationPatient, dtype=float)
        position = np.asarray(ds.ImagePositionPatient, dtype=float)
        row_cosine = orientation[:3]
        col_cosine = orientation[3:]
        normal = np.cross(row_cosine, col_cosine)
        location = float(np.dot(position, normal))
    except Exception:
        location = safe_float(getattr(ds, "SliceLocation", 0.0), 0.0)
    instance = safe_float(getattr(ds, "InstanceNumber", 0.0), 0.0)
    return location, instance, str(fallback_path)


def discover_lctsc_subjects(root: Path) -> List[SubjectInfo]:
    if not root.exists():
        raise FileNotFoundError(f"DATA_ROOT does not exist: {root}")

    logging.info("Scanning DICOM files under: %s", root)
    ct_groups: Dict[Tuple[str, str], List[Tuple[Path, pydicom.dataset.Dataset]]] = {}
    rt_records: List[Tuple[str, Path, Set[str]]] = []
    unreadable = 0

    for dirpath, _, filenames in os.walk(root):
        for filename in filenames:
            path = Path(dirpath) / filename
            try:
                ds = pydicom.dcmread(
                    str(path),
                    stop_before_pixels=True,
                    force=True,
                    specific_tags=[
                        "Modality",
                        "PatientID",
                        "SeriesInstanceUID",
                        "SOPInstanceUID",
                        "ImagePositionPatient",
                        "ImageOrientationPatient",
                        "SliceLocation",
                        "InstanceNumber",
                        "ReferencedFrameOfReferenceSequence",
                    ],
                )
            except Exception:
                unreadable += 1
                continue

            modality = str(getattr(ds, "Modality", "")).upper()
            patient_id = str(getattr(ds, "PatientID", path.parents[2].name if len(path.parents) > 2 else path.parent.name))

            if modality == "CT":
                series_uid = str(getattr(ds, "SeriesInstanceUID", ""))
                if series_uid:
                    ct_groups.setdefault((patient_id, series_uid), []).append((path, ds))
            elif modality == "RTSTRUCT":
                rt_records.append((patient_id, path, get_rt_referenced_series_uids(ds)))

    logging.info(
        "Found %d CT series and %d RTSTRUCT files. Unreadable/non-DICOM files skipped: %d",
        len(ct_groups),
        len(rt_records),
        unreadable,
    )

    subjects: List[SubjectInfo] = []
    for (patient_id, series_uid), records in ct_groups.items():
        if len(records) < MIN_CT_SLICES:
            continue

        records.sort(key=lambda item: ct_sort_key(item[1], item[0]))
        ct_files = [item[0] for item in records]

        matching_rt: Optional[Path] = None
        # Prefer an explicit SeriesInstanceUID reference.
        for rt_patient, rt_path, referenced_uids in rt_records:
            if series_uid in referenced_uids:
                matching_rt = rt_path
                break
        # Fall back to PatientID when the nested reference is absent.
        if matching_rt is None:
            same_patient = [r[1] for r in rt_records if r[0] == patient_id]
            if len(same_patient) == 1:
                matching_rt = same_patient[0]

        subjects.append(
            SubjectInfo(
                patient_id=patient_id,
                ct_series_uid=series_uid,
                ct_files=ct_files,
                rtstruct_path=matching_rt,
            )
        )

    subjects.sort(key=lambda s: s.patient_id)
    if SUBJECT_FILTER:
        subjects = [s for s in subjects if SUBJECT_FILTER.lower() in s.patient_id.lower()]
    if MAX_SUBJECTS is not None:
        subjects = subjects[:MAX_SUBJECTS]

    if not subjects:
        raise RuntimeError(
            "No usable CT series were found. Check DATA_ROOT and MIN_CT_SLICES."
        )

    logging.info("Subjects selected for this run: %d", len(subjects))
    for subject in subjects:
        logging.info(
            "  %s | %d CT slices | RTSTRUCT: %s",
            subject.patient_id,
            len(subject.ct_files),
            subject.rtstruct_path if subject.rtstruct_path else "not found",
        )
    return subjects


def read_ct_series(ct_files: Sequence[Path]) -> sitk.Image:
    reader = sitk.ImageSeriesReader()
    reader.SetFileNames([str(p) for p in ct_files])
    reader.MetaDataDictionaryArrayUpdateOn()
    reader.LoadPrivateTagsOn()
    image = reader.Execute()
    return sitk.Cast(image, sitk.sitkFloat32)


def numpy_mask_to_sitk(mask_zyx: np.ndarray, reference: sitk.Image) -> sitk.Image:
    image = sitk.GetImageFromArray(mask_zyx.astype(np.uint8, copy=False))
    image.CopyInformation(reference)
    return image


def choose_lung_roi_names(roi_names: Sequence[str]) -> Tuple[str, str]:
    normalized = {name: normalize_roi_name(name) for name in roi_names}

    def score(name: str, side: str) -> int:
        n = normalized[name]
        if "lung" not in n:
            return -100
        side_tokens = (
            ["left", "lungl", "ltlung", "lunglt"]
            if side == "left"
            else ["right", "lungr", "rtlung", "lungrt"]
        )
        value = 10
        if any(token in n for token in side_tokens):
            value += 20
        if "gtv" in n or "tumor" in n or "tumour" in n:
            value -= 20
        return value

    left_scores = sorted(((score(n, "left"), n) for n in roi_names), reverse=True)
    right_scores = sorted(((score(n, "right"), n) for n in roi_names), reverse=True)
    if not left_scores or left_scores[0][0] < 20:
        raise RuntimeError(f"Could not identify a left-lung ROI. Available ROIs: {list(roi_names)}")
    if not right_scores or right_scores[0][0] < 20:
        raise RuntimeError(f"Could not identify a right-lung ROI. Available ROIs: {list(roi_names)}")
    return left_scores[0][1], right_scores[0][1]


def prepare_rtutils_series_directory(ct_files: Sequence[Path], temp_root: Path) -> Path:
    parents = {p.parent.resolve() for p in ct_files}
    if len(parents) == 1:
        # Confirm that this folder does not contain a second CT series.
        return next(iter(parents))

    series_dir = temp_root / "ct_series"
    series_dir.mkdir(parents=True, exist_ok=True)
    for idx, src in enumerate(ct_files):
        suffix = src.suffix if src.suffix else ".dcm"
        dst = series_dir / f"slice_{idx:05d}{suffix}"
        try:
            os.link(src, dst)
        except Exception:
            shutil.copy2(src, dst)
    return series_dir


def load_reference_masks_rtstruct(
    subject: SubjectInfo,
    original_ct_image: sitk.Image,
    processed_ct_image: sitk.Image,
) -> Tuple[np.ndarray, np.ndarray, Tuple[str, str]]:
    if subject.rtstruct_path is None:
        raise FileNotFoundError("No RTSTRUCT was matched to this CT series.")
    if RTStructBuilder is None:
        raise ImportError(
            "rt-utils is not installed. Install it with: python -m pip install rt-utils"
        )

    with tempfile.TemporaryDirectory(prefix="lctsc_rtutils_") as tmp:
        ct_dir = prepare_rtutils_series_directory(subject.ct_files, Path(tmp))
        rtstruct = RTStructBuilder.create_from(
            dicom_series_path=str(ct_dir),
            rt_struct_path=str(subject.rtstruct_path),
        )
        roi_names = rtstruct.get_roi_names()
        left_name, right_name = choose_lung_roi_names(roi_names)
        logging.info("  Reference ROIs: left='%s', right='%s'", left_name, right_name)

        left_yxz = rtstruct.get_roi_mask_by_name(left_name)
        right_yxz = rtstruct.get_roi_mask_by_name(right_name)

    # rt-utils returns rows x columns x slices.
    left_zyx = np.transpose(np.asarray(left_yxz, dtype=bool), (2, 0, 1))
    right_zyx = np.transpose(np.asarray(right_yxz, dtype=bool), (2, 0, 1))

    expected_original_shape = tuple(reversed(original_ct_image.GetSize()))
    if left_zyx.shape != expected_original_shape:
        raise RuntimeError(
            f"RTSTRUCT mask shape {left_zyx.shape} does not match original CT shape "
            f"{expected_original_shape}. Check the matched series."
        )

    left_img = numpy_mask_to_sitk(left_zyx, original_ct_image)
    right_img = numpy_mask_to_sitk(right_zyx, original_ct_image)

    if processed_ct_image.GetSize() != original_ct_image.GetSize() or processed_ct_image.GetSpacing() != original_ct_image.GetSpacing():
        left_img = sitk.Resample(
            left_img,
            processed_ct_image,
            sitk.Transform(),
            sitk.sitkNearestNeighbor,
            0,
            sitk.sitkUInt8,
        )
        right_img = sitk.Resample(
            right_img,
            processed_ct_image,
            sitk.Transform(),
            sitk.sitkNearestNeighbor,
            0,
            sitk.sitkUInt8,
        )

    left = sitk.GetArrayFromImage(left_img) > 0
    right = sitk.GetArrayFromImage(right_img) > 0
    return left, right, (left_name, right_name)


def save_mask(mask: np.ndarray, reference: sitk.Image, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = numpy_mask_to_sitk(mask, reference)
    sitk.WriteImage(image, str(path), useCompression=NIFTI_COMPRESSION)
