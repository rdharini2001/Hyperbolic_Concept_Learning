"""Patient-level cross-validation folds.

All nodules of a patient go to the same fold, so no patient appears in both
training and evaluation. Stratification uses a nodule-level column (default:
reader-mean malignancy >= 3) so folds have similar risk mix, which matters
later for the risk head even though Part 2 does not use malignancy.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold


def make_patient_folds(
    df: pd.DataFrame, n_folds: int = 5, seed: int = 0, stratify: np.ndarray | None = None
) -> np.ndarray:
    groups = df["patient_id"].to_numpy()
    folds = np.full(len(df), -1, dtype=int)
    if stratify is not None:
        splitter = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        it = splitter.split(np.zeros(len(df)), stratify, groups)
    else:
        it = GroupKFold(n_splits=n_folds).split(np.zeros(len(df)), groups=groups)
    for k, (_, test_idx) in enumerate(it):
        folds[test_idx] = k
    return folds


def split_indices(folds: np.ndarray, test_fold: int, n_folds: int) -> dict[str, np.ndarray]:
    """test = ``test_fold``, val = next fold, train = the rest."""
    if not 0 <= test_fold < n_folds:
        raise ValueError(f"test_fold must be in [0, {n_folds})")
    val_fold = (test_fold + 1) % n_folds
    return {
        "train": np.where((folds != test_fold) & (folds != val_fold))[0],
        "val": np.where(folds == val_fold)[0],
        "test": np.where(folds == test_fold)[0],
    }


def assert_no_patient_leakage(df: pd.DataFrame, splits: dict[str, np.ndarray]) -> None:
    names = list(splits)
    pats = {k: set(df["patient_id"].iloc[v]) for k, v in splits.items()}
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            shared = pats[a] & pats[b]
            if shared:
                raise RuntimeError(f"Patient leakage between {a} and {b}: {sorted(shared)[:5]} ...")
