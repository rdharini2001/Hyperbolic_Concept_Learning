"""Train the concept space for one (geometry, fold, seed) and export out-of-fold concept scores.

    hypertrace-train                                                  # packaged configs/train.yaml
    hypertrace-train --set geometry.name=euclidean data.test_fold=3 seed=2
    hypertrace-train --config configs/experiments/fmcib.yaml --set geometry.name=euclidean
    hypertrace-train --print-config                                   # show the resolved config and exit

On SLURM, slurm/train_array.sbatch maps the array index onto (geometry, fold, seed).
The run directory is deterministic
(<output_root>/runs/<experiment>_<geometry>_<feature>_d<dim>[_t<tag>]_f<fold>_s<seed>), so a requeued
job resumes from checkpoints/last.pt instead of starting over. A run folder that already holds a run
with a different config is refused (set ``tag=...`` or another output_root).

Outputs in the run directory:
    config.yaml               fully resolved config + run metadata (command, git commit, versions)
    metrics_per_epoch.csv     training curves
    metrics.json              train, val and test metrics of the best checkpoint, plus the linear-probe
                              diagnostics (test/probe_embedding/..., test/probe_input/...)
    predictions_test.csv      per-nodule concept probabilities + soft labels (input to Part 3)
    concept_space.pt          positions of every concept node, geometry, curvature (product: per-factor
                              layout under "factors"), for figures / Part 3
    feature_norm.pt           training-split mean / SD used to standardise the inputs (if data.standardize)
    checkpoints/{best,last}.pt
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from hypertrace.concepts import ConceptHierarchy
from hypertrace.concepts.ontology import load_concept_tree
from hypertrace.config import ConfigError, default_config, load_config, load_yaml, run_metadata, save_config
from hypertrace.data.dataset import ConceptData
from hypertrace.eval.probes import probe_aurocs
from hypertrace.geometry import curvature_of
from hypertrace.models.model import ConceptSpaceModel, build_model
from hypertrace.training import evaluate, fit, load_weights, predict, resolve_device, set_seed

log = logging.getLogger("hypertrace.train")


def run_name(cfg: dict) -> str:
    d = cfg["data"]
    dim = cfg["model"]["projector"]["embed_dim"]
    tag = f"_t{cfg['tag']}" if cfg.get("tag") not in (None, "") else ""
    geo = cfg["geometry"]["name"]
    if geo == "product":  # the assignment is part of the arm: product-matched, product-swapped, ...
        geo = f"product-{cfg['geometry']['product']['assignment']}"
    return (f"{cfg['experiment']}_{geo}_{d['image_feature']}_d{dim}{tag}"
            f"_f{d['test_fold']}_s{cfg['seed']}")


# Settings that may differ between a run folder's saved config and a resumed job.
_RESUME_FREE = {("paths",), ("trainer", "device"), ("diagnostics",), ("_meta",), ("concepts",)}


def _comparable(cfg: dict, prefix: tuple = ()) -> dict:
    out = {}
    for k, v in cfg.items():
        key = (*prefix, k)
        if key in _RESUME_FREE:
            continue
        out[k] = _comparable(v, key) if isinstance(v, dict) else v
    return out


def _diff(a: dict, b: dict, prefix: str = "") -> list[str]:
    keys = sorted(set(a) | set(b))
    out = []
    for k in keys:
        va, vb = a.get(k, "<missing>"), b.get(k, "<missing>")
        if isinstance(va, dict) and isinstance(vb, dict):
            out += _diff(va, vb, f"{prefix}{k}.")
        elif va != vb:
            out.append(f"{prefix}{k}: {va!r} -> {vb!r}")
    return out


def check_run_dir(run_dir: Path, cfg: dict) -> None:
    """Refuse to resume a run folder that was trained with different settings (it would silently
    mix two configs, or return an old finished run under the new settings)."""
    old_path = run_dir / "config.yaml"
    if not (old_path.exists() and (run_dir / "checkpoints" / "last.pt").exists()):
        return
    old = load_yaml(old_path)
    changes = _diff(_comparable(old), _comparable(cfg))
    if changes:
        raise ConfigError(f"{run_dir} already holds a run with different settings ({'; '.join(changes[:6])}"
                          f"{' ...' if len(changes) > 6 else ''}). Set tag=<name> for a new run folder, use "
                          "another paths.output_root, or delete the folder.")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=Path, default=default_config("train"))
    p.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="override config values")
    p.add_argument("--run-dir", type=Path, default=None, help="default: <output_root>/runs/<run name>")
    p.add_argument("--print-config", action="store_true")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Console entry point (returns None so the process exits with status 0)."""
    try:
        run(argv)
    except ConfigError as e:
        raise SystemExit(f"config error: {e}") from None


def run(argv: list[str] | None = None) -> dict[str, float]:
    """Train one run; returns the test metrics (used by tests and notebooks)."""
    a = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = load_config(a.config, a.set)
    if a.print_config:
        import yaml
        print(yaml.safe_dump(cfg, sort_keys=False))
        return {}

    run_dir = a.run_dir or Path(cfg["paths"]["output_root"]) / "runs" / run_name(cfg)
    check_run_dir(run_dir, cfg)
    run_dir.mkdir(parents=True, exist_ok=True)
    tree = load_concept_tree(cfg["concepts_file"])
    save_config({**cfg, "concepts": tree}, run_dir / "config.yaml", run_metadata(argv))
    log.info("Run directory: %s", run_dir)

    t = cfg["trainer"]
    set_seed(int(cfg["seed"]), bool(t.get("deterministic", True)))
    torch.set_float32_matmul_precision("high")
    device = resolve_device(t.get("device", "auto"))

    hierarchy = ConceptHierarchy.from_config(tree)
    data = ConceptData(manifest_path=cfg["paths"]["manifest"], cache_dir=cfg["paths"]["cache_dir"],
                       hierarchy=hierarchy, **cfg["data"])
    m = cfg["model"]
    concepts = m.get("concepts", {})
    needs_text = concepts.get("mode") == "text" or concepts.get("init") == "text"
    if needs_text and data.text_latents is None:
        raise ValueError(f"model.concepts needs prompt embeddings, but {data.text_problem}. "
                         "Use model.concepts.mode=free model.concepts.init=tree.")
    if m["projector"]["share_projection"] and data.image_dim != data.text_dim:
        raise ValueError(
            f"share_projection=true but image_feature '{cfg['data']['image_feature']}' has dim {data.image_dim} "
            f"!= text dim {data.text_dim}. Use --set model.projector.share_projection=false.")
    model = build_model(cfg, hierarchy, data, pos_weight=data.pos_weight()).to(device)
    with torch.no_grad():  # reference: how much of the tree the initial points already encode
        init_h = model.tree_metrics()

    thr = cfg["data"]["binary_threshold"]
    train_eval = data.loader("train") if t.get("eval_train", True) else None
    res = fit(model, data.loader("train", shuffle=True), data.loader("val"), cfg["optim"], t, run_dir, device, thr,
              train_eval_loader=train_eval)

    load_weights(model, res.best_path, device)
    train_m = evaluate(model, data.loader("train"), device, "train", thr)
    val = evaluate(model, data.loader("val"), device, "val", thr)
    test = evaluate(model, data.loader("test"), device, "test", thr)
    diag = run_diagnostics(cfg, model, data, device)
    for stage, block in (("val", val), ("test", test)):
        for kind, res_k in diag.items():
            block.update({f"{stage}/probe_{kind}/{k}": v for k, v in res_k.get(stage, {}).items()})
    _export(model, data, device, run_dir)
    (run_dir / "metrics.json").write_text(json.dumps({
        "best_ckpt": str(res.best_path), "best_epoch": res.best_epoch, "epochs_run": res.epochs_run,
        "stopped_early": res.stopped_early, "select_metric": res.select_metric,
        "init": {f"init/{k}": v for k, v in init_h.items()},
        "train": train_m, "val": val, "test": test}, indent=2))
    extra = "".join(f" test/probe_{k}={test[f'test/probe_{k}/auroc_macro']:.3f}" for k in diag
                    if f"test/probe_{k}/auroc_macro" in test)
    log.info("train/auroc_macro=%.3f val/auroc_macro=%.3f test/auroc_macro=%.3f test/edge_in_cone=%.3f%s",
             train_m["train/auroc_macro"], val["val/auroc_macro"], test["test/auroc_macro"],
             test["test/edge_in_cone"], extra)
    return test


@torch.no_grad()
def lesion_tangent(model: ConceptSpaceModel, x: torch.Tensor, device: torch.device) -> np.ndarray:
    """Learned lesion embeddings in tangent coordinates at the origin (log_0), eval mode."""
    model.eval()
    z = torch.cat([model.projector.embed_image(xb.to(device)).cpu() for xb in x.split(512)])
    return model.geometry.to_tangent(z).numpy()


def _input_probe_key(cfg: dict) -> str:
    d = cfg["data"]
    tree = json.dumps(load_concept_tree(cfg["concepts_file"]), sort_keys=True).encode()
    key = json.dumps({"cache": str(Path(cfg["paths"]["cache_dir"]).resolve()),
                      "manifest": str(Path(cfg["paths"]["manifest"]).resolve()),
                      "data": {k: d.get(k) for k in ("image_feature", "standardize", "pca_dim", "test_fold",
                                                     "n_folds", "binary_threshold", "min_occupancy")},
                      "C": cfg.get("diagnostics", {}).get("probe_C", 0.1),
                      "tree": hashlib.sha1(tree).hexdigest()}, sort_keys=True)
    return hashlib.sha1(key.encode()).hexdigest()[:16]


def run_diagnostics(cfg: dict, model: ConceptSpaceModel, data: ConceptData, device: torch.device) -> dict:
    """Linear probes on the input features and on the learned embedding (hypertrace/eval/probes.py).
    The input probe does not depend on geometry, dimension or seed: cached per fold in
    <output_root>/reference/ and reused by every run of that fold."""
    dg = {"embedding_probe": True, "input_probe": True, "probe_C": 0.1, **(cfg.get("diagnostics") or {})}
    soft, thr, C = data.soft.numpy(), cfg["data"]["binary_threshold"], float(dg["probe_C"])
    out: dict[str, dict] = {}
    if dg["embedding_probe"]:
        out["embedding"] = probe_aurocs(lesion_tangent(model, data.x, device), soft, data.splits,
                                        data.concept_names, thr, C)
    if dg["input_probe"]:
        ref = Path(cfg["paths"]["output_root"]) / "reference" / f"input_probe_{_input_probe_key(cfg)}.json"
        if ref.exists():
            out["input"] = json.loads(ref.read_text())
        else:
            log.info("Input-feature probe for fold %s (cached in %s)", cfg["data"]["test_fold"], ref)
            out["input"] = probe_aurocs(data.x.numpy(), soft, data.splits, data.concept_names, thr, C)
            ref.parent.mkdir(parents=True, exist_ok=True)
            tmp = ref.with_suffix(f".tmp{os.getpid()}")
            tmp.write_text(json.dumps(out["input"]))
            os.replace(tmp, ref)
    return out


def _export(model: ConceptSpaceModel, data: ConceptData, device: torch.device, run_dir: Path) -> None:
    p = predict(model, data.loader("test"), device)
    rows, names = p["row"], data.concept_names
    out = pd.concat([
        data.df.iloc[rows][["nodule_id", "patient_id", "fold"]].reset_index(drop=True),
        pd.DataFrame(p["probs"], columns=[f"prob_{n}" for n in names]),
        pd.DataFrame(data.soft[rows].numpy(), columns=[f"soft_{n}" for n in names]),
        pd.DataFrame({"lesion_depth": p["lesion_depth"]}),
    ], axis=1)
    out.to_csv(run_dir / "predictions_test.csv", index=False)
    if data.feature_mean is not None or data.pca_components is not None:
        torch.save({"mean": data.feature_mean, "std": data.feature_std, "pca_mean": data.pca_mean,
                    "pca_components": data.pca_components}, run_dir / "feature_norm.pt")
    with torch.no_grad():
        nodes = model.concept_points().cpu()
        torch.save({
            "names": data.hierarchy.names,
            "parents": [n.parent for n in data.hierarchy.nodes],
            "scored": data.hierarchy.scored_names,
            "points": nodes,
            "depth": model.geometry.depth(nodes).cpu(),
            "geometry": model.geometry.name,
            "curvature": float(model.geometry.c) if hasattr(model.geometry, "c") else None,  # Poincaré c (v2)
            "kappa": None if model.geometry.name == "product" else curvature_of(model.geometry),  # signed
            "factors": model.factor_info() if hasattr(model, "factor_info") else None,
        }, run_dir / "concept_space.pt")


if __name__ == "__main__":
    main()
