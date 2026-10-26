#!/usr/bin/env python
"""Pick the (fold, seed) to draw for each dimension, by VALIDATION AUROC only (never test).

    python scripts/best_runs.py --runs $HT_OUTPUT_ROOT/runs --tag v3 [--dims "10 20 40"] [--geometries "..."]

For each embedding dimension it prints "<dim> <fold> <seed>": the split where the arms drawn together
have the highest mean validation macro-AUROC, so that every arm in a figure shows the same nodules.
Only (fold, seed) pairs that finished for ALL requested arms are considered.
"""
import argparse
import json
import re
from collections import defaultdict
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--runs", required=True)
    p.add_argument("--tag", default="")
    p.add_argument("--dims", default="")
    p.add_argument("--n-arms", type=int, default=0, help="require this many arms per (dim, fold, seed)")
    a = p.parse_args()
    want = set(a.dims.split())
    suffix = f"_t{a.tag}" if a.tag else ""
    pat = re.compile(rf"_d(\d+){re.escape(suffix)}_f(\d+)_s(\d+)$")
    table = defaultdict(dict)  # (dim, fold, seed) -> {arm: val auroc}
    for f in sorted(Path(a.runs).glob("*/metrics.json")):
        m = pat.search(f.parent.name)
        if not m or (want and m.group(1) not in want):
            continue
        if not a.tag and re.search(r"_t[A-Za-z0-9.-]+_f\d+_s\d+$", f.parent.name):
            continue  # untagged request: skip tagged runs
        v = json.loads(f.read_text()).get("val", {}).get("val/auroc_macro")
        if v is not None:
            table[tuple(map(int, m.groups()))][f.parent.name[: m.start()]] = v
    best = {}
    for (dim, fold, seed), arms in table.items():
        if a.n_arms and len(arms) < a.n_arms:
            continue
        score = sum(arms.values()) / len(arms)
        key = (len(arms), score)
        if dim not in best or key > best[dim][0]:
            best[dim] = (key, fold, seed)
    for dim in sorted(best):
        print(dim, best[dim][1], best[dim][2])


if __name__ == "__main__":
    main()