"""Check what the frozen readouts carry before training anything.

    hypertrace-zeroshot --cache-dir $HT_CACHE_ROOT/ctclip_scan_dil0 --manifest $HT_DATA_ROOT/manifest_folds.csv
    hypertrace-zeroshot ... --probe          # also patient-grouped linear probes

Two numbers per concept and feature:

``zeroshot`` AUROC of cos(lesion latent, concept text latent) in CT-CLIP's own space
             (only for features in the image-text space: masked_latent, global_latent).
             Near 0.5 everywhere means CT-CLIP's text space doesn't see these concepts
             zero-shot. That is common and not fatal, because the projector is trained.
Baselines (rows ``diameter_only`` and, with ``--radiomics``, ``radiomics``): what a
single measurement / 21 hand-crafted features achieve. A learned feature that does not
beat them carries no extra concept information.

``probe``    AUROC of a logistic-regression probe with patient-grouped CV. This is the
             ceiling of what a linear readout of that feature can learn, so it picks the
             cache (input_mode, dilate_tokens) and the image_feature to train with.

Why compare features: CT-CLIP's image projection is position-specific (a linear map
over the flattened token grid). In ``scan`` mode a nodule covers only a few tokens,
so ``masked_latent`` also encodes *where* the nodule is. ``token_mean`` averages the
nodule's own tokens and is position-invariant. The probe shows which matters more.
"""

from __future__ import annotations

import argparse
from importlib.resources import files
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from hypertrace.concepts import ConceptHierarchy, LabelBuilder
from hypertrace.concepts.ontology import load_concept_tree
from hypertrace.data.dataset import prompt_ensemble
from hypertrace.data.embedding_cache import EmbeddingCache
from hypertrace.data.manifest import load_manifest

CLIP_SPACE = ("masked_latent", "global_latent")


def _auroc(y: np.ndarray, s: np.ndarray) -> float:
    m = ~np.isnan(y)
    return float(roc_auc_score(y[m], s[m])) if m.any() and 0 < y[m].sum() < m.sum() else float("nan")


def _probe(X: np.ndarray, y: np.ndarray, groups: np.ndarray, folds: int) -> float:
    m = ~np.isnan(y)
    clf = make_pipeline(SimpleImputer(), StandardScaler(), LogisticRegression(max_iter=3000, C=0.1))
    try:
        pr = cross_val_predict(clf, X[m], y[m], groups=groups[m], cv=GroupKFold(folds), method="predict_proba")[:, 1]
        return _auroc(y[m], pr)
    except ValueError:  # a training fold had a single class (rare concept, small data)
        return float("nan")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cache-dir", required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--concepts", default=str(files("hypertrace") / "configs" / "labels" / "lidc.yaml"),
                   help="label mapping (v3: configs/labels/*.yaml) or self-contained tree (configs/concepts/*.yaml)")
    p.add_argument("--features", nargs="+", default=None,
                   help="readouts to test (default: every vector readout in the cache)")
    p.add_argument("--probe", action="store_true", help="add patient-grouped linear-probe AUROCs")
    p.add_argument("--radiomics", default=None, help="CSV from hypertrace-radiomics: adds a 'radiomics' baseline row")
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--out", default=None, help="CSV path (default: <cache-dir>/zeroshot.csv)")
    a = p.parse_args(argv)

    hierarchy = ConceptHierarchy.from_config(load_concept_tree(a.concepts))
    cache = EmbeddingCache(a.cache_dir)
    df = load_manifest(a.manifest)
    imgs = cache.load_images(df["nodule_id"].tolist())
    feats = a.features or [k for k, v in imgs.items() if v.ndim == 2]
    labels = LabelBuilder(hierarchy).build(df)
    groups = df["patient_id"].to_numpy()

    nodes = None
    if cache.has_text():
        text = cache.read_text()
        if text["node_names"] == hierarchy.names:
            nodes = prompt_ensemble(text["latents"], text["owner"], len(hierarchy))[hierarchy.scored_indices]
        else:
            print("note: text.pt was encoded for another concept tree; zero-shot column skipped "
                  "(hypertrace-extract --set only_text=true cache_dir=... re-encodes it)")

    # Baselines that need no encoder: the Part-1 diameter, and simple radiomics if given.
    inputs: dict[str, np.ndarray] = {f: imgs[f].float().numpy() for f in feats}
    baselines = []
    if "diameter_mm" in df.columns:
        inputs["diameter_only"] = df[["diameter_mm"]].to_numpy(dtype=np.float64)
        baselines.append("diameter_only")
    if a.radiomics:
        rad = pd.read_csv(a.radiomics).set_index("nodule_id")
        missing = set(df["nodule_id"]) - set(rad.index)
        if missing:
            raise SystemExit(f"{len(missing)} nodules missing from {a.radiomics} (e.g. {sorted(missing)[:3]})")
        inputs["radiomics"] = rad.loc[df["nodule_id"]].to_numpy(dtype=np.float64)
        baselines.append("radiomics")

    rows = []
    for feat, X in inputs.items():
        sims = None
        if nodes is not None and feat in CLIP_SPACE:
            sims = (torch.nn.functional.normalize(torch.from_numpy(X).float(), dim=-1) @ nodes.T).numpy()
        for k, name in enumerate(labels.names):
            y = labels.binary[:, k]
            row = {"feature": feat, "concept": name, "prevalence": float(np.nanmean(y)),
                   "zeroshot": _auroc(y, sims[:, k]) if sims is not None else float("nan")}
            if feat == "diameter_only":  # one number: its AUROC directly (sign-free: max of both directions)
                auc = _auroc(y, X[:, 0])
                row["probe"] = max(auc, 1 - auc) if np.isfinite(auc) else auc
            elif a.probe or feat in baselines:
                row["probe"] = _probe(X, y, groups, a.folds)
            rows.append(row)
    res = pd.DataFrame(rows)
    out = a.out or str(Path(a.cache_dir) / "zeroshot.csv")
    res.to_csv(out, index=False)
    for metric in ("zeroshot", "probe"):
        if metric in res and res[metric].notna().any():
            print(f"\n{metric} AUROC")
            print(res.pivot(index="concept", columns="feature", values=metric).round(3).to_string())
    if "probe" in res:
        print("\nmean probe AUROC per feature (diameter_only = AUROC of the diameter itself, either direction):")
        print(res.groupby("feature")["probe"].mean().round(3).sort_values(ascending=False).to_string())
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
