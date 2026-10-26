"""Add a patient-level ``fold`` column to the Part-1 manifest.

    hypertrace-make-folds --manifest data/manifest.csv --out data/manifest_folds.csv --n-folds 5

Folds are stratified by reader-mean malignancy >= 3 when ``malignancy_r*``
columns exist, so every fold has a similar risk mix for the later risk head.
Run this once and commit/keep the output; every arm and seed must reuse it.
"""

from __future__ import annotations

import argparse
import logging

import numpy as np

from hypertrace.data.folds import make_patient_folds
from hypertrace.data.manifest import load_manifest

log = logging.getLogger("hypertrace.make_folds")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--manifest", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--n-folds", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--stratify-attribute", default="malignancy")
    p.add_argument("--stratify-threshold", type=float, default=3.0)
    a = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO)

    df = load_manifest(a.manifest)
    cols = [c for c in df.columns if c.startswith(f"{a.stratify_attribute}_r")]
    strat = None
    if cols:
        mean = df[cols].mean(axis=1, skipna=True).to_numpy()
        strat = (np.nan_to_num(mean, nan=0.0) >= a.stratify_threshold).astype(int)
        log.info("Stratifying on %s (positives: %d / %d)", cols, strat.sum(), len(strat))
    df["fold"] = make_patient_folds(df, a.n_folds, a.seed, strat)
    # load_manifest made crop paths absolute, which avoids cwd surprises inside SLURM jobs.
    df.to_csv(a.out, index=False)
    log.info("Wrote %s; nodules per fold: %s", a.out, df["fold"].value_counts().sort_index().to_dict())


if __name__ == "__main__":
    main()
