"""Build the Part-2 manifest from the LIDC-IDRI bucket metadata.

    hypertrace-build-manifest --bucket-dir $HT_BUCKET --out $HT_DATA_ROOT/manifest.csv
    hypertrace-build-manifest --bucket-dir $HT_BUCKET --inspect      # dump headers/values

Only the CSVs are read here (annotations.csv, nodules.csv, scans.csv). Image and
mask paths are written relative to the bucket root and resolved at extraction time.

Output: one row per nodule, with
    patient_id, nodule_id (``<case>_n<k>``), case_id, instance_label (value in masks_instance),
    image_path, instance_mask_path, spacing_z/y/x, diameter_mm, consensus_voxels, ...,
    <attr>_r1 ... <attr>_rK   per-reader LIDC codes from the SR *_term columns
"""

from __future__ import annotations

import argparse
import logging
import re
from importlib.resources import files
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

log = logging.getLogger("hypertrace.build_manifest")

DEFAULT_MAP = files("hypertrace") / "configs" / "bucket" / "lidc_hf.yaml"
_LEADING_INT = re.compile(r"^\s*(\d+)\s+out of\s+\d+")


class MappingError(ValueError):
    pass


def _require(df: pd.DataFrame, cols: list[str], where: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise MappingError(
            f"{where}: columns {missing} not found. Available: {list(df.columns)}\n"
            "Fix configs/bucket/lidc_hf.yaml (or pass --map); --inspect shows the real headers."
        )


def parse_term(values: pd.Series, attr: str, value_map: dict[str, float] | None) -> pd.Series:
    """SR term -> LIDC integer code. 'N out of 5 (...)' -> N; named terms via ``value_map``; NaN stays NaN."""
    value_map = value_map or {}
    out = pd.Series(float("nan"), index=values.index)
    present = values.notna()
    as_str = values[present].astype(str).str.strip()
    named = as_str.map(value_map)
    scaled = as_str.str.extract(_LEADING_INT)[0].astype(float)
    out.loc[as_str.index] = named.fillna(scaled)
    bad = as_str[out.loc[as_str.index].isna()]
    if len(bad):
        raise MappingError(
            f"'{attr}': unrecognised SR terms {sorted(bad.unique())[:10]}. Add them to value_maps.{attr}."
        )
    return out


def build_manifest(bucket_dir: Path, cfg: dict[str, Any]) -> pd.DataFrame:
    a_cfg, n_cfg, s_cfg, f_cfg = cfg["annotations"], cfg["nodules"], cfg["scans"], cfg.get("filters", {})
    ann = pd.read_csv(bucket_dir / a_cfg["file"])
    nod = pd.read_csv(bucket_dir / n_cfg["file"])
    scans = pd.read_csv(bucket_dir / s_cfg["file"])
    key = list(n_cfg["nodule_key"])
    chars: dict[str, str] = dict(a_cfg["characteristics"])
    checks: dict[str, str] = dict(a_cfg.get("check_scores") or {})
    _require(ann, [*a_cfg["nodule_key"], a_cfg["reader_order"], *chars.values()], a_cfg["file"])
    _require(nod, [*key, n_cfg["instance_label"], *n_cfg.get("carry", [])], n_cfg["file"])
    _require(scans, [s_cfg["case_id"], *s_cfg.get("carry", [])], s_cfg["file"])
    if nod.duplicated(key).any():
        raise MappingError(f"nodules.csv key {key} is not unique")

    # ---- ratings: SR terms -> LIDC codes, cross-checked against *_score where present
    ann = ann.rename(columns=dict(zip(a_cfg["nodule_key"], key, strict=True)))
    vmaps = a_cfg.get("value_maps") or {}
    for attr, col in chars.items():
        ann[attr] = parse_term(ann[col], attr, vmaps.get(attr))
        sc = checks.get(attr)
        if sc and sc in ann:
            both = ann[sc].notna()
            disagree = both & (ann.loc[both, attr] != ann.loc[both, sc])
            if disagree.any():
                raise MappingError(f"'{attr}': {int(disagree.sum())} terms disagree with {sc}; check value_maps")

    # ---- long (one row per annotation) -> wide reader columns, in reader_order
    ann = ann.sort_values([*key, a_cfg["reader_order"]])
    ann["_reader"] = ann.groupby(key).cumcount() + 1
    wide = ann.pivot_table(index=key, columns="_reader", values=list(chars), aggfunc="first", dropna=False)
    wide.columns = [f"{attr}_r{k}" for attr, k in wide.columns]
    wide = wide.reset_index()
    n_read = ann.groupby(key).size()
    if n_read.max() > 4:
        log.info("%d nodules have > 4 annotations (up to %d; LIDC had 4 readers, the clustering merged some). "
                 "All are averaged.", int((n_read > 4).sum()), int(n_read.max()))

    cols = list(dict.fromkeys([*key, *n_cfg.get("carry", [])]))
    df = nod[cols].assign(instance_label=nod[n_cfg["instance_label"]].astype(int))
    df = df.merge(wide, on=key, how="left", validate="one_to_one")
    if df.filter(regex=r"_r1$").isna().all(axis=1).any():
        raise MappingError("Some nodules have no annotation rows; nodule_key mismatch between CSVs")

    # ---- filters
    n0 = len(df)
    if f_cfg.get("default_gt_only", True):
        df = df[df["in_default_gt"].astype(bool)]
    log.info("Nodules: %d total, %d kept by default-GT rule (>= 3 annotations)", n0, len(df))
    sc = scans.rename(columns={s_cfg["case_id"]: "case_id"})[["case_id", *s_cfg.get("carry", [])]]
    df = df.merge(sc, on="case_id", how="left", validate="many_to_one")
    spread = f_cfg.get("max_z_gap_spread_mm")
    if spread is not None:
        irregular = (df["z_gap_max"] - df["z_gap_min"]) > float(spread)
        log.info("Dropping %d nodules on %d scans with irregular slice gaps (> %.2f mm spread)",
                 int(irregular.sum()), df.loc[irregular, "case_id"].nunique(), float(spread))
        df = df[~irregular]

    # ---- ids and paths
    df["nodule_id"] = df["case_id"].astype(str) + "_n" + df["instance_label"].astype(int).astype(str)
    p = cfg["paths"]
    df["image_path"] = [p["image"].format(**r) for r in df.to_dict("records")]
    df["instance_mask_path"] = [p["instance_mask"].format(**r) for r in df.to_dict("records")]
    if df["nodule_id"].duplicated().any():
        raise MappingError("nodule_id is not unique")
    front = ["patient_id", "nodule_id", "case_id", "instance_label", "image_path", "instance_mask_path",
             "spacing_z", "spacing_y", "spacing_x", "diameter_mm", "consensus_voxels", "n_annotations"]
    front = [c for c in front if c in df]
    return df[front + [c for c in df.columns if c not in front]].reset_index(drop=True)


def inspect(bucket_dir: Path, cfg: dict[str, Any]) -> None:
    for section in ("annotations", "nodules", "scans"):
        path = bucket_dir / cfg[section]["file"]
        if not path.exists():
            print(f"\n## {path.name}: NOT FOUND in {bucket_dir}")
            continue
        df = pd.read_csv(path)
        print(f"\n## {path.name}  ({len(df)} rows, {df.shape[1]} columns)")
        for c in df.columns:
            u = df[c].dropna().unique()
            sample = ", ".join(map(str, u[:6])) + (" ..." if len(u) > 6 else "")
            print(f"  {c:28s} {str(df[c].dtype):8s} null={df[c].isna().mean():5.1%} n_unique={len(u):<6d} "
                  f"e.g. {sample[:80]}")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bucket-dir", required=True, type=Path, help="bucket root (or a folder with the 3 CSVs)")
    ap.add_argument("--map", type=Path, default=None, help=f"column map YAML (default: {DEFAULT_MAP})")
    ap.add_argument("--out", type=Path)
    ap.add_argument("--inspect", action="store_true")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    cfg = yaml.safe_load(a.map.read_text() if a.map else DEFAULT_MAP.read_text())
    if a.inspect:
        inspect(a.bucket_dir, cfg)
        return
    if a.out is None:
        ap.error("--out is required unless --inspect")
    df = build_manifest(a.bucket_dir, cfg)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(a.out, index=False)
    readers = df.filter(regex=r"^spiculation_r\d+$").notna().sum(axis=1).value_counts().sort_index().to_dict()
    log.info("Wrote %s: %d nodules, %d scans, %d patients; annotations per nodule: %s",
             a.out, len(df), df["case_id"].nunique(), df["patient_id"].nunique(), readers)
    missing = [q for q in df["image_path"].unique() if not (a.bucket_dir / q).exists()]
    if missing:
        log.info("%d of %d scan images not found under %s (fine if you only have the CSVs here; "
                 "set paths.bucket_root to the full download for extraction).",
                 len(missing), df["case_id"].nunique(), a.bucket_dir)


if __name__ == "__main__":
    main()
