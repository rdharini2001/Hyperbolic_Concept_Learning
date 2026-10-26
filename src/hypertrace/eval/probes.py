"""Linear probes on one fold's split: where does the concept bottleneck lose information?

Three numbers per run, all fitted on the training split and scored on val/test:

``input``      logistic probe on the input features (standardised / PCA as in training): the ceiling
               for anything built on these features with this split.
``embedding``  the same probe on the learned lesion embedding, in tangent coordinates at the origin
               (log_0; the identity for the Euclidean twin).
``head``       the model's own distance-to-concept scores (test/auroc_macro).

embedding ≈ input > head  -> the projection keeps the information, the distance head loses it.
embedding ≈ head < input  -> the projection itself is the bottleneck.

The probe is the one used by ``hypertrace-zeroshot --probe`` (standardise, L2 logistic regression,
C = 0.1), on binarised soft labels (soft >= threshold); nodules without a label are skipped.
"""

from __future__ import annotations

import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def probe_aurocs(x: np.ndarray, soft: np.ndarray, splits: dict[str, np.ndarray], names: list[str],
                 threshold: float = 0.5, C: float = 0.1, eval_splits: tuple[str, ...] = ("val", "test"),
                 ) -> dict[str, dict[str, float]]:
    """{split: {"auroc/<concept>": value, "auroc_macro": value}} for each split in ``eval_splits``.

    x: (N, D) features of every nodule; soft: (N, K) soft labels; splits: index arrays (train, val, test).
    """
    y_all = np.where(np.isnan(soft), np.nan, (soft >= threshold).astype(float))
    tr = splits["train"]
    out: dict[str, dict[str, float]] = {s: {} for s in eval_splits}
    for k, name in enumerate(names):
        ytr = y_all[tr, k]
        ok = ~np.isnan(ytr)
        if len(np.unique(ytr[ok])) < 2:
            continue
        clf = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=3000))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            clf.fit(x[tr][ok], ytr[ok])
        for s in eval_splits:
            idx = splits[s]
            y = y_all[idx, k]
            m = ~np.isnan(y)
            if len(np.unique(y[m])) < 2:
                continue
            out[s][f"auroc/{name}"] = float(roc_auc_score(y[m], clf.predict_proba(x[idx][m])[:, 1]))
    for s in eval_splits:
        vals = list(out[s].values())
        out[s]["auroc_macro"] = float(np.mean(vals)) if vals else float("nan")
    return out
