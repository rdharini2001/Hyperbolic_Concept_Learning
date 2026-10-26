"""Part-2 manifest: one row per nodule (built by ``hypertrace-build-manifest``).

Required columns
    patient_id           LIDC patient id; folds group on it (some patients have 2 scans)
    nodule_id            unique, ``<case_id>_n<instance_label>``
    case_id              bucket scan id; extraction loads each scan once
    instance_label       value of this nodule in ``masks_instance/<case>.nii.gz``
    image_path           scan, relative to the bucket root (or absolute)
    instance_mask_path   instance mask, relative to the bucket root (or absolute)

Also used when present: ``spacing_z/y/x`` (cross-checked against the NIfTI header),
``consensus_voxels`` (cross-checked against the instance mask), ``diameter_mm``,
per-reader ratings ``<attr>_r<k>``, and ``fold`` (added by ``hypertrace-make-folds``).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

REQUIRED = ("patient_id", "nodule_id", "case_id", "instance_label", "image_path", "instance_mask_path")


def load_manifest(path: str | Path, bucket_root: str | Path | None = None) -> pd.DataFrame:
    """Read and validate; with ``bucket_root``, image/mask paths are made absolute."""
    path = Path(path).expanduser()
    df = pd.read_csv(path) if path.suffix == ".csv" else pd.read_parquet(path)
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise KeyError(f"Manifest {path} is missing required columns {missing}. "
                       "Build it with `hypertrace-build-manifest`.")
    if df["nodule_id"].duplicated().any():
        dups = df.loc[df["nodule_id"].duplicated(), "nodule_id"].head().tolist()
        raise ValueError(f"Duplicate nodule_id values, e.g. {dups}")
    df = df.copy()
    for c in ("patient_id", "nodule_id", "case_id"):
        df[c] = df[c].astype(str)
    df["instance_label"] = df["instance_label"].astype(int)
    if bucket_root is not None:
        root = Path(bucket_root).expanduser()
        for c in ("image_path", "instance_mask_path"):
            df[c] = [str(p if Path(p).is_absolute() else root / p) for p in df[c]]
    return df.reset_index(drop=True)
