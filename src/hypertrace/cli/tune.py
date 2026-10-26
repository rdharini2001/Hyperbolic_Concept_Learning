"""Tune training settings with the same budget for every geometry; select on validation only.

    hypertrace-tune --config configs/experiments/fmcib.yaml --dims 16       # 16 trials x 2 geometries x 5 folds
    hypertrace-tune ... --shard 0/2  and  --shard 1/2                       # split the runs over two GPUs
    hypertrace-tune ... --report-only                                       # just aggregate finished runs

The search space is a YAML file (default: configs/tune/default.yaml). Trials are drawn once, with a
fixed seed, and every geometry x dimension x fold x seed runs the SAME trials, so no arm gets more
tuning than another. Trial 0 is the base config unchanged. Each trial trains in its own run folder
(``tag=<name><trial>``); finished runs are skipped, so the command can be re-run after a pre-emption.

Selection never looks at test metrics:

``per_fold``  (reported first, the clean estimate) for each geometry, dimension and outer test fold, the
              trial with the best mean validation score (over seeds) on THAT fold's validation split;
              its test score is reported. Settings may differ between folds.
``global``    one trial per geometry and dimension, best mean validation score over all folds. Convenient
              for a final sweep (prints the --set string), but a fold's validation split is another fold's
              test split, so test scores under this choice are slightly optimistic. Use per_fold numbers
              in the paper.

Writes to --out (default <output_root>/tuning/<name>): trials.csv, runs.csv, selected_per_fold.csv,
selected_global.csv.
"""

from __future__ import annotations

import argparse
import itertools
import json
import logging
import random
from pathlib import Path

import pandas as pd

from hypertrace.config import default_config, load_config, load_yaml

log = logging.getLogger("hypertrace.tune")


def draw_trials(space: dict) -> list[dict]:
    """[{key: value}] with trial 0 = {} (base config). Deterministic for a given space file."""
    params = space["params"]
    keys = list(params)
    combos = [dict(zip(keys, vals, strict=True)) for vals in itertools.product(*(params[k] for k in keys))]
    if space.get("sample", "random") == "grid":
        chosen = combos
    else:
        rng = random.Random(int(space.get("seed", 0)))
        chosen = rng.sample(combos, min(int(space["n_trials"]) - 1, len(combos)))
    return [{}, *chosen]


def _fmt(v) -> str:
    return json.dumps(v)  # null / true / false / numbers, as YAML reads them


def run_jobs(args: argparse.Namespace, trials: list[dict], name: str) -> list[dict]:
    jobs = []
    for d, g, t, f, s in itertools.product(args.dims, args.geometries, range(len(trials)), args.folds, args.seeds):
        jobs.append({"dim": d, "geometry": g, "trial": t, "fold": f, "seed": s, "tag": f"{name}{t:02d}"})
    return jobs


def geometry_sets(geometry: str) -> list[str]:
    """``product:matched`` -> geometry.name=product + the assignment; anything else -> geometry.name."""
    if geometry.startswith("product:"):
        return ["geometry.name=product", f"geometry.product.assignment={geometry.split(':', 1)[1]}"]
    return [f"geometry.name={geometry}"]


def _argv(args: argparse.Namespace, job: dict, trial: dict) -> list[str]:
    sets = [*args.set, *geometry_sets(job["geometry"]), f"model.projector.embed_dim={job['dim']}",
            f"data.test_fold={job['fold']}", f"seed={job['seed']}", f"tag={job['tag']}",
            *[f"{k}={_fmt(v)}" for k, v in trial.items()]]
    return ["--config", str(args.config), "--set", *sets]


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=Path, default=default_config("train"))
    p.add_argument("--space", type=Path, default=default_config("tune/default"))
    p.add_argument("--geometries", nargs="+", default=["poincare", "euclidean"],
                   help="poincare euclidean sphere kstereo product:<assignment> (e.g. product:matched)")
    p.add_argument("--dims", nargs="+", type=int, default=[16])
    p.add_argument("--folds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    p.add_argument("--seeds", nargs="+", type=int, default=[0])
    p.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="overrides for every run")
    p.add_argument("--select-metric", default="val/auroc_macro", help="validation metric to maximise")
    p.add_argument("--shard", default="0/1", help="i/n: run every n-th job starting at i (parallel workers)")
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--report-only", action="store_true")
    a = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    if not a.select_metric.startswith("val/"):
        raise SystemExit("--select-metric must be a validation metric (val/...): tuning never looks at test")

    space = load_yaml(a.space)
    name = space.get("name", "tune")
    trials = draw_trials(space)
    cfg = load_config(a.config, a.set)
    out = a.out or Path(cfg["paths"]["output_root"]) / "tuning" / name
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{"trial": i, "tag": f"{name}{i:02d}", **{k: _fmt(v) for k, v in t.items()}}
                  for i, t in enumerate(trials)]).to_csv(out / "trials.csv", index=False)
    jobs = run_jobs(a, trials, name)

    from hypertrace.cli import train  # imported late: --help stays fast

    if not a.report_only:
        i, n = (int(v) for v in a.shard.split("/"))
        mine = jobs[i::n]
        log.info("%d trials x %d geometries x %d dims x %d folds x %d seeds = %d runs (this worker: %d)",
                 len(trials), len(a.geometries), len(a.dims), len(a.folds), len(a.seeds), len(jobs), len(mine))
        for k, job in enumerate(mine):
            log.info("[%d/%d] %s", k + 1, len(mine), job)
            try:
                train.run(_argv(a, job, trials[job["trial"]]))
            except Exception as e:  # one failed setting must not stop the search
                log.error("run failed: %s: %s", job, e)

    rows = []
    for job in jobs:
        run_cfg = load_config(a.config, _argv(a, job, trials[job["trial"]])[3:])
        mfile = Path(run_cfg["paths"]["output_root"]) / "runs" / train.run_name(run_cfg) / "metrics.json"
        if not mfile.exists():
            continue
        m = json.loads(mfile.read_text())
        rows.append({**job, "val": m["val"].get(a.select_metric), "test_auroc_macro": m["test"]["test/auroc_macro"],
                     "train_auroc_macro": m.get("train", {}).get("train/auroc_macro"),
                     "test_tree_map": m["test"].get("test/tree_map"),
                     "test_parent_child_violation": m["test"].get("test/parent_child_violation")})
    if not rows:
        log.warning("no finished runs yet under %s", cfg["paths"]["output_root"])
        return
    df = pd.DataFrame(rows)
    df.to_csv(out / "runs.csv", index=False)
    missing = len(jobs) - len(df)
    if missing:
        log.warning("%d of %d runs not finished: selections below use what exists", missing, len(jobs))

    # per-fold selection (clean): best trial on each fold's own validation split
    per_seed = df.groupby(["geometry", "dim", "fold", "trial"], as_index=False)[
        ["val", "test_auroc_macro", "train_auroc_macro", "test_tree_map", "test_parent_child_violation"]].mean()
    best = per_seed.loc[per_seed.groupby(["geometry", "dim", "fold"])["val"].idxmax()]
    best.to_csv(out / "selected_per_fold.csv", index=False)
    base = per_seed[per_seed.trial == 0].set_index(["geometry", "dim", "fold"])["test_auroc_macro"]
    summary = best.groupby(["geometry", "dim"]).agg(
        test_auroc=("test_auroc_macro", "mean"), test_sd=("test_auroc_macro", "std"),
        train_auroc=("train_auroc_macro", "mean"), tree_map=("test_tree_map", "mean"),
        trials=("trial", lambda s: " ".join(map(str, s))))
    summary["base_test_auroc"] = base.groupby(["geometry", "dim"]).mean()
    print(f"\nPer-fold selection on {a.select_metric} (test never used for choosing):")
    print(summary.round(3).to_string())

    # global selection (convenience for a final sweep)
    glob = per_seed.groupby(["geometry", "dim", "trial"], as_index=False)[["val", "test_auroc_macro"]].mean()
    gbest = glob.loc[glob.groupby(["geometry", "dim"])["val"].idxmax()]
    gbest = gbest.assign(set=[" ".join(f"{k}={_fmt(v)}" for k, v in trials[t].items()) or "(base config)"
                              for t in gbest.trial])
    gbest.to_csv(out / "selected_global.csv", index=False)
    print("\nGlobal choice per geometry and dimension (for TRAIN_SET in a final sweep; slightly optimistic test):")
    print(gbest.round(3).to_string(index=False))
    print(f"\nTables in {out}")


if __name__ == "__main__":
    main()
