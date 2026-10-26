"""Turn multi-reader LIDC ratings into concept labels.

Two views are produced for every scored concept:

* ``soft``   in [0, 1], used as the training target (BCE accepts soft targets).
             This keeps reader disagreement in the label instead of discarding it.
* ``binary`` = soft >= ``binary_threshold``, used for AUROC / F1 reporting.

Missing values (no reader rated it, or no diameter) become NaN and are masked
out of the loss and metrics.

Label spec types (set per node in ``configs/concepts/*.yaml``):

``ordinal_readers``      mean over readers of (score - low) / (high - low); ``invert`` flips it.
                         With low=1, high=5 and threshold 0.5, "binary" means reader-mean >= 3.
``categorical_readers``  fraction of readers whose code is in ``positive_values``
                         (e.g. LIDC calcification 1-5 = present, 6 = absent).
``threshold``            1 if ``column`` >= ``threshold`` else 0 (e.g. diameter_mm); ``invert`` flips it.
``range``                1 if ``low`` <= ``column`` < ``high`` (either bound may be omitted), e.g. size bins.

Reader columns are found with ``reader_column_pattern`` (default
``{attribute}_r{reader}``, e.g. ``spiculation_r1 ... spiculation_r4``).

LIDC rating scales (pylidc docs), for reference:
    spiculation, lobulation  1 = none ... 5 = marked
    margin                   1 = poorly defined ... 5 = sharp
    texture                  1 = non-solid/GGO, 3 = part-solid, 5 = solid (2 and 4 in between)
    sphericity               1 = linear, 3 = ovoid, 5 = round
    calcification            1 popcorn, 2 laminated, 3 solid, 4 non-central, 5 central, 6 absent
    malignancy               1 = highly unlikely ... 5 = highly suspicious (not used in Part 2)
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from hypertrace.concepts.hierarchy import ConceptHierarchy


@dataclass
class ConceptLabels:
    names: list[str]
    soft: np.ndarray  # (N, K) float32, NaN = missing
    binary: np.ndarray  # (N, K) float32 in {0, 1}, NaN = missing

    @property
    def mask(self) -> np.ndarray:
        return ~np.isnan(self.soft)


class LabelBuilder:
    def __init__(
        self,
        hierarchy: ConceptHierarchy,
        binary_threshold: float = 0.5,
        reader_column_pattern: str = "{attribute}_r{reader}",
    ) -> None:
        self.hierarchy = hierarchy
        self.binary_threshold = binary_threshold
        self.pattern = reader_column_pattern

    def reader_columns(self, df: pd.DataFrame, attribute: str) -> list[str]:
        regex = re.compile(
            "^" + re.escape(self.pattern).replace(re.escape("{attribute}"), re.escape(attribute))
            .replace(re.escape("{reader}"), r"\d+") + "$"
        )
        cols = sorted(c for c in df.columns if regex.match(c))
        if not cols:
            raise KeyError(
                f"No reader columns for attribute '{attribute}' "
                f"(pattern '{self.pattern}'). Available: {list(df.columns)}"
            )
        return cols

    def _one(self, df: pd.DataFrame, spec: Mapping[str, Any]) -> np.ndarray:
        kind = spec["type"]
        if kind == "ordinal_readers":
            r = df[self.reader_columns(df, spec["attribute"])].to_numpy(dtype=np.float64)
            low, high = float(spec.get("low", 1)), float(spec.get("high", 5))
            with np.errstate(invalid="ignore"), _quiet_nanmean():
                v = (np.nanmean(r, axis=1) - low) / (high - low)
            v = np.clip(v, 0.0, 1.0)
        elif kind == "categorical_readers":
            r = df[self.reader_columns(df, spec["attribute"])].to_numpy(dtype=np.float64)
            pos = np.isin(r, np.asarray(spec["positive_values"], dtype=np.float64))
            rated = ~np.isnan(r)
            n = rated.sum(axis=1)
            with np.errstate(invalid="ignore", divide="ignore"):
                v = np.where(n > 0, (pos & rated).sum(axis=1) / n, np.nan)
        elif kind == "threshold":
            col = df[spec["column"]].to_numpy(dtype=np.float64)
            v = np.where(np.isnan(col), np.nan, (col >= float(spec["threshold"])).astype(np.float64))
        elif kind == "range":
            col = df[spec["column"]].to_numpy(dtype=np.float64)
            lo, hi = float(spec.get("low", -np.inf)), float(spec.get("high", np.inf))
            v = np.where(np.isnan(col), np.nan, ((col >= lo) & (col < hi)).astype(np.float64))
        else:
            raise ValueError(f"Unknown label type '{kind}'")
        if spec.get("invert", False):
            v = 1.0 - v
        return v.astype(np.float32)

    def build(self, df: pd.DataFrame) -> ConceptLabels:
        names, cols = [], []
        for node in self.hierarchy.nodes:
            if node.label is None:
                continue
            names.append(node.name)
            cols.append(self._one(df, node.label))
        soft = np.stack(cols, axis=1)
        binary = np.where(np.isnan(soft), np.nan, (soft >= self.binary_threshold).astype(np.float32))
        return ConceptLabels(names=names, soft=soft, binary=binary.astype(np.float32))


class _quiet_nanmean:
    """Silence 'Mean of empty slice' for nodules no reader rated (result is NaN, as intended)."""

    def __enter__(self):
        import warnings

        self._ctx = warnings.catch_warnings()
        self._ctx.__enter__()
        warnings.simplefilter("ignore", category=RuntimeWarning)

    def __exit__(self, *exc):
        return self._ctx.__exit__(*exc)
