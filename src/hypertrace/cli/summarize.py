"""Aggregate metrics.json files across runs into mean +/- SD per arm.

    hypertrace-summarize --runs $HT_OUTPUT_ROOT/runs --out summary.csv

Runs are grouped by everything in ``run_name`` except fold and seed, so each
row is one arm (e.g. main_poincare_token_mean_d64) over 5 folds x N seeds.
``--per-concept`` adds a per-concept AUROC table (mean over runs).

Diagnostic columns (runs trained after the training-fix patch):
    train_auroc_macro           the same model on its own training split (overfitting = train - test)
    probe_embedding/auroc_macro logistic probe on the learned lesion embedding
    probe_input/auroc_macro     logistic probe on the input features, same split (the ceiling)
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd

DEFAULT_KEYS = ["auroc_macro", "auprc_macro", "f1_macro", "brier_macro",
                "parent_child_violation", "exclusive_violation",
                "edge_in_cone", "non_edge_in_cone", "depth_spearman", "tree_map", "dist_spearman",
                "probe_embedding/auroc_macro", "probe_input/auroc_macro"]


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs", required=True)
    p.add_argument("--split", default="test", choices=["val", "test"])
    p.add_argument("--out", default=None)
    p.add_argument("--per-concept", action="store_true")
    p.add_argument("--groups", action="store_true",
                   help="per concept group: AUROC, exclusivity, ordinal and hierarchy metrics (mean over runs)")
    a = p.parse_args(argv)

    rows = []
    for f in sorted(Path(a.runs).glob("*/metrics.json")):
        full = json.loads(f.read_text())
        m = full[a.split]
        train_auroc = full.get("train", {}).get("train/auroc_macro")
        name = f.parent.name
        arm = re.sub(r"_f\d+_s\d+$", "", name)
        fold = re.search(r"_f(\d+)_s\d+$", name)
        seed = re.search(r"_s(\d+)$", name)
        per = {k.split("/", 2)[2]: v for k, v in m.items() if k.startswith(f"{a.split}/auroc/")}  # head only
        grp = {k.split("/", 1)[1]: v for k, v in m.items() if k.startswith(f"{a.split}/group/")}
        rows.append({"arm": arm, "fold": int(fold.group(1)) if fold else -1,
                     "seed": int(seed.group(1)) if seed else -1,
                     "train_auroc_macro": train_auroc,
                     **{k: m.get(f"{a.split}/{k}") for k in DEFAULT_KEYS},
                     **{f"auroc/{k}": v for k, v in per.items()}, **grp})
    if not rows:
        raise SystemExit(f"No metrics.json found under {a.runs}")
    df = pd.DataFrame(rows)
    keys = [k for k in ["train_auroc_macro", *DEFAULT_KEYS] if df[k].notna().any()]
    agg = df.groupby("arm")[keys].agg(["mean", "std"]).round(3)
    agg.insert(0, ("n_runs", ""), df.groupby("arm").size())
    print(agg.to_string())
    if a.per_concept:
        cols = [c for c in df.columns if c.startswith("auroc/")]
        print("\nper-concept AUROC (mean over runs)")
        print(df.groupby("arm")[cols].mean().T.round(3).to_string())
    if a.groups:
        cols = [c for c in df.columns if c.startswith("group/")]
        if cols:
            g = df.groupby("arm")[cols].mean().T
            g.index = pd.MultiIndex.from_tuples([tuple(c.split("/")[1:3]) for c in g.index], names=["group", "metric"])
            print("\nper concept group (mean over runs; violation = lower is better)")
            print(g.sort_index().round(3).to_string())
        else:
            print("\nno group metrics in these runs (trained before the per-group metrics existed)")
    if a.out:
        df.to_csv(a.out, index=False)
        print(f"\nPer-run table written to {a.out}")


if __name__ == "__main__":
    main()
