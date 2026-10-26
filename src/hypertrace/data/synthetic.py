"""Synthetic data in the exact layout and schema of Dharini24/LIDC-IDRI-bucket.

    python -m hypertrace.data.synthetic --out /scratch/$USER/ht_synth_bucket --patients 40

Writes ``images/<case>.nii.gz`` (int16 HU, anisotropic spacing, -2048 out-of-FOV
edge), ``masks_instance/<case>.nii.gz`` (uint16 nodule ids), and ``annotations.csv``,
``nodules.csv`` and ``scans.csv`` with the real column names. Ratings are SR terms,
including the named categories that leave ``*_score`` empty. Nodules are spheres,
and ratings loosely track size, so a model can learn *something*. Not for real results.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

SCALE5 = ("subtlety", "lobulation", "spiculation", "malignancy")
NAMED = {
    "margin": {1: "Indistinct margin", 5: "Circumscribed margin"},
    "texture": {1: "non-solid pulmonary nodule", 3: "part-solid pulmonary nodule", 5: "solid pulmonary nodule"},
    "sphericity": {1: "linear", 3: "ovoid", 5: "round"},
}
CALC = {1: "Popcorn calcification sign", 3: "Solid appearance", 6: "Absent"}


def _term(attr: str, v: int) -> tuple[str, float]:
    if attr in NAMED and v in NAMED[attr]:
        return NAMED[attr][v], float("nan")  # named concept: *_score left empty, as in the bucket
    return f"{v} out of 5", float(v)


def make_synthetic_bucket(out: str | Path, n_patients: int = 30, shape=(40, 72, 72), seed: int = 0) -> Path:
    import nibabel as nib

    rng = np.random.default_rng(seed)
    out = Path(out)
    for d in ("images", "masks_instance"):
        (out / d).mkdir(parents=True, exist_ok=True)
    Z, Y, X = shape
    zz, yy, xx = np.meshgrid(np.arange(Z), np.arange(Y), np.arange(X), indexing="ij")
    ann_rows, nod_rows, scan_rows = [], [], []
    for p in range(n_patients):
        pid = f"LIDC-IDRI-{p + 1:04d}"
        case = f"{pid}_{rng.integers(1e9, 1e10)}"
        sz, sxy = float(rng.choice([1.25, 2.0, 2.5])), float(rng.uniform(0.6, 0.85))
        ct = rng.normal(-800, 40, shape)
        ct[:, :, :3] = -2048
        inst = np.zeros(shape, np.uint16)
        n_nod = int(rng.integers(1, 4))
        for k in range(1, n_nod + 1):
            for _ in range(20):  # re-draw until the sphere doesn't vanish inside an earlier nodule
                r_mm = float(rng.uniform(2.5, 9.0))
                c = (rng.uniform(0.15, 0.85) * Z, rng.uniform(0.3, 0.7) * Y, rng.uniform(0.3, 0.7) * X)
                d = np.sqrt(((zz - c[0]) * sz) ** 2 + ((yy - c[1]) * sxy) ** 2 + ((xx - c[2]) * sxy) ** 2)
                sel = (d <= r_mm) & (inst == 0)
                if sel.sum() >= 0.5 * (d <= r_mm).sum():
                    break
            else:
                continue
            inst[sel] = k
            ct[sel] = rng.normal(40, 20, int(sel.sum()))
            n_ann = int(rng.choice([2, 3, 4, 4]))
            big = (r_mm - 2.5) / 6.5
            nod_rows.append({
                "case_id": case, "patient_id": pid, "nodule_id": k, "tracking_uid": f"2.25.{p}{k}",
                "n_annotations": n_ann, "in_default_gt": n_ann >= 3, "consensus_voxels": int(sel.sum()),
                "volume_mm3": float(sel.sum() * sz * sxy * sxy), "diameter_mm": 2 * r_mm,
                "centroid_z": c[0], "centroid_y": c[1], "centroid_x": c[2],
            })
            for a in range(1, n_ann + 1):
                row = {"case_id": case, "patient_id": pid, "nodule_id": k, "annotation_idx": a, "has_sr": True}
                for attr in (*SCALE5, "margin", "texture", "sphericity"):
                    base = 1 + 4 * (1 - big if attr == "margin" else big)
                    v = int(np.clip(round(base + rng.normal(0, 0.8)), 1, 5))
                    row[f"{attr}_term"], row[f"{attr}_score"] = _term(attr, v)
                row["calcification_term"] = CALC[6 if rng.random() > 0.15 else int(rng.choice([1, 3]))]
                row["calcification_score"] = float("nan")
                row["internal_structure_term"], row["internal_structure_score"] = "Soft tissue", float("nan")
                ann_rows.append(row)
        # RAS affine with the anisotropic spacing; array stored (x, y, z) as NIfTI expects
        aff = np.diag([sxy, sxy, sz, 1.0])
        nib.save(nib.Nifti1Image(ct.astype(np.int16).transpose(2, 1, 0), aff), out / "images" / f"{case}.nii.gz")
        nib.save(nib.Nifti1Image(inst.transpose(2, 1, 0), aff), out / "masks_instance" / f"{case}.nii.gz")
        scan_rows.append({"case_id": case, "patient_id": pid, "n_slices": Z, "spacing_x": sxy, "spacing_y": sxy,
                          "spacing_z": sz, "z_gap_min": sz, "z_gap_max": sz, "manufacturer": "SYNTH",
                          "in_luna16": False, "luna16_subset": None})
    pd.DataFrame(ann_rows).to_csv(out / "annotations.csv", index=False)
    pd.DataFrame(nod_rows).to_csv(out / "nodules.csv", index=False)
    pd.DataFrame(scan_rows).to_csv(out / "scans.csv", index=False)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--patients", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    print(make_synthetic_bucket(a.out, a.patients, seed=a.seed))
