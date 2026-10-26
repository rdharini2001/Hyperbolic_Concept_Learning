"""Load bucket scans and instance masks in one canonical orientation.

The bucket stores ``images/<case>.nii.gz`` (int16 HU, native geometry) and
``masks_instance/<case>.nii.gz`` (uint16, one id per clustered nodule at 50 %
consensus). Both are reoriented to the closest canonical orientation (RAS+) and
returned as (z, y, x) arrays with (z, y, x) spacing in mm. Every scan then has the
same anatomical axis order, whatever the scanner wrote.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class Volume:
    array: np.ndarray  # (z, y, x)
    spacing: tuple[float, float, float]  # (z, y, x) mm


def load_volume(path: str | Path, dtype: type | None = None) -> Volume:
    try:
        import nibabel as nib
    except ImportError as e:  # pragma: no cover
        raise ImportError("Reading NIfTI needs nibabel: `pip install nibabel`") from e
    img = nib.as_closest_canonical(nib.load(str(path)))
    arr = np.asarray(img.dataobj)  # applies scl_slope/scl_inter when present
    if arr.ndim != 3:
        raise ValueError(f"{path}: expected a 3D volume, got shape {arr.shape}")
    if dtype is not None:
        arr = arr.astype(dtype, copy=False)
    sx, sy, sz = (float(v) for v in img.header.get_zooms()[:3])
    return Volume(array=np.ascontiguousarray(arr.transpose(2, 1, 0)), spacing=(sz, sy, sx))


def check_spacing(vol: Volume, expected: tuple[float, float, float] | None, name: str, atol: float = 0.02) -> None:
    """Header spacing must match the manifest (scans.csv) spacing; a mismatch means a wrong file or axis order."""
    if expected is None or any(np.isnan(expected)):
        return
    if not np.allclose(vol.spacing, expected, atol=atol):
        raise ValueError(f"{name}: NIfTI spacing (z, y, x) {vol.spacing} != manifest spacing {expected}")
