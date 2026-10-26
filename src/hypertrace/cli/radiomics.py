"""Hand-crafted nodule features: the simple baseline every learned feature has to beat.

    hypertrace-radiomics                       # $HT_DATA_ROOT/manifest_folds.csv -> $HT_CACHE_ROOT/radiomics.csv
    hypertrace-zeroshot ... --probe --radiomics $HT_CACHE_ROOT/radiomics.csv

CPU only, from the raw HU scan and the consensus nodule mask at native spacing (no
resampling). 21 features, chosen to map onto the concepts:

size        volume_mm3, eq_diameter_mm, diameter_mm (Part 1), extent_z/y/x_mm
shape       sphericity (pi^(1/3) (6V)^(2/3) / A), elongation and flatness (PCA of voxel coordinates),
            surface_mm2 (exposed voxel faces)
attenuation hu_mean, hu_std, hu_p10, hu_p25, hu_p50, hu_p75, hu_p90,
            frac_ggo (-750 to -300 HU), frac_calcified (> 200 HU)
margin      hu_shell (mean HU in a 2-voxel shell outside the mask), contrast (hu_mean - hu_shell)

These are deliberately simple (no pyradiomics dependency). The mask is the readers'
consensus, so for size and shape this baseline sees what the readers drew.
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage

from hypertrace.cli.extract import _check_label
from hypertrace.data.manifest import load_manifest
from hypertrace.data.volumes import load_volume

log = logging.getLogger("hypertrace.radiomics")


def nodule_features(hu: np.ndarray, mask: np.ndarray, spacing: tuple[float, float, float]) -> dict[str, float]:
    """hu, mask: (z, y, x) arrays at native spacing (mm)."""
    sz, sy, sx = spacing
    idx = np.argwhere(mask)
    lo, hi = np.maximum(idx.min(0) - 3, 0), np.minimum(idx.max(0) + 4, mask.shape)
    sl = tuple(slice(a, b) for a, b in zip(lo, hi, strict=True))
    m, h = mask[sl], hu[sl].astype(np.float64)
    vals = h[m]
    vol = m.sum() * sz * sy * sx
    # surface: exposed faces along each axis, weighted by face area
    pad = np.pad(m, 1)
    faces = [np.abs(np.diff(pad.astype(np.int8), axis=a)).sum() for a in range(3)]
    area = faces[0] * sy * sx + faces[1] * sz * sx + faces[2] * sz * sy
    coords = idx * np.array(spacing)
    if len(coords) >= 3:
        ev = np.sort(np.linalg.eigvalsh(np.cov(coords.T)))[::-1].clip(min=0)
        elong = float(np.sqrt(ev[1] / ev[0])) if ev[0] > 0 else 1.0
        flat = float(np.sqrt(ev[2] / ev[0])) if ev[0] > 0 else 1.0
    else:
        elong = flat = 1.0
    shell = ndimage.binary_dilation(m, iterations=2) & ~m
    hu_shell = float(h[shell].mean()) if shell.any() else np.nan
    p = np.percentile(vals, [10, 25, 50, 75, 90])
    ext = (idx.max(0) - idx.min(0) + 1) * np.array(spacing)
    return {
        "volume_mm3": vol, "eq_diameter_mm": (6 * vol / np.pi) ** (1 / 3),
        "extent_z_mm": ext[0], "extent_y_mm": ext[1], "extent_x_mm": ext[2],
        "surface_mm2": area, "sphericity": np.pi ** (1 / 3) * (6 * vol) ** (2 / 3) / max(area, 1e-6),
        "elongation": elong, "flatness": flat,
        "hu_mean": vals.mean(), "hu_std": vals.std(), "hu_p10": p[0], "hu_p25": p[1], "hu_p50": p[2],
        "hu_p75": p[3], "hu_p90": p[4],
        "frac_ggo": float(((vals > -750) & (vals < -300)).mean()), "frac_calcified": float((vals > 200).mean()),
        "hu_shell": hu_shell, "contrast": vals.mean() - hu_shell,
    }


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--manifest", default=f"{os.environ.get('HT_DATA_ROOT', './data')}/manifest_folds.csv")
    p.add_argument("--bucket-dir", default=os.environ.get("HT_BUCKET", "./bucket"))
    p.add_argument("--out", default=f"{os.environ.get('HT_CACHE_ROOT', './cache')}/radiomics.csv")
    p.add_argument("--label-size-tolerance", type=float, default=0.05)
    a = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    df = load_manifest(a.manifest, a.bucket_dir)
    rows = []
    for i, (_, grp) in enumerate(df.groupby("case_id", sort=True)):
        r0 = grp.iloc[0]
        img, inst = load_volume(r0["image_path"]), load_volume(r0["instance_mask_path"])
        for row in grp.itertuples():
            mask = _check_label(inst.array, int(row.instance_label), getattr(row, "consensus_voxels", None),
                                row.nodule_id, a.label_size_tolerance)
            feats = nodule_features(img.array, mask, tuple(img.spacing))
            rows.append({"nodule_id": row.nodule_id, "diameter_mm": getattr(row, "diameter_mm", np.nan), **feats})
        if (i + 1) % 50 == 0:
            log.info("%d scans, %d nodules", i + 1, len(rows))
    out = pd.DataFrame(rows)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(a.out, index=False)
    log.info("Wrote %d nodules x %d features to %s", len(out), out.shape[1] - 1, a.out)


if __name__ == "__main__":
    main()
