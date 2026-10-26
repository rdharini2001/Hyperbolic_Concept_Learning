"""Concept ontology + per-dataset label mapping -> the concept tree the model uses.

Two kinds of concept files are accepted by ``load_concept_tree``:

* a **label mapping** (v3, ``configs/labels/<dataset>.yaml``)::

      name: lidc
      ontology: ../ontology/lung_nodule.yaml     # relative to this file
      labels: {size_lt6: {type: range, ...}, ...}

  The ontology holds the clinical tree (names, parents, display terms, RadLex IDs, Lung-RADS
  roles, prompts); the mapping only says which nodes this dataset labels and how. Every
  ontology node stays in the tree; nodes without a label are unscored.

* a **self-contained tree** (v1/v2, ``configs/concepts/*.yaml``): nodes carry their own
  ``label`` specs. Returned unchanged.

The result is always the self-contained form ``{name, nodes: [{name, parent, prompts, label?,
exclusive, display?, radlex?, lungrads?}]}`` plus ``ontology`` / ``dataset`` provenance, so the
rest of the code (and every saved run config) sees one format.
"""

from __future__ import annotations

import copy
from pathlib import Path

import yaml


class OntologyError(ValueError):
    pass


def _read(path: Path) -> dict:
    if not path.exists():
        raise OntologyError(f"Concept file not found: {path}")
    data = yaml.safe_load(path.read_text())
    if not isinstance(data, dict):
        raise OntologyError(f"{path}: top level must be a mapping")
    return data


def resolve_concept_tree(mapping: dict, here: Path) -> dict:
    """Apply a label mapping (already loaded) to its ontology, which is read relative to ``here``."""
    if "ontology" not in mapping:
        return mapping
    onto_path = (here / mapping["ontology"]).resolve()
    onto = _read(onto_path)
    nodes = copy.deepcopy(onto.get("nodes") or [])
    names = [n["name"] for n in nodes]
    labels = mapping.get("labels") or {}
    unknown = sorted(set(labels) - set(names))
    if unknown:
        raise OntologyError(f"Label mapping '{mapping.get('name')}' labels nodes that are not in the ontology "
                            f"{onto_path.name}: {unknown}")
    for n in nodes:
        if "label" in n:
            raise OntologyError(f"Ontology node '{n['name']}' carries a label rule; labels belong in the "
                                "dataset mapping file")
        if n["name"] in labels:
            n["label"] = dict(labels[n["name"]])
    return {
        "name": f"{onto.get('name', onto_path.stem)}+{mapping.get('name', 'labels')}",
        "ontology": {"name": onto.get("name"), "version": onto.get("version"), "file": onto_path.name},
        "dataset": mapping.get("name"),
        "nodes": nodes,
    }


def load_concept_tree(path: str | Path) -> dict:
    """The concept tree described by ``path`` (label mapping or self-contained tree), resolved."""
    path = Path(path)
    return resolve_concept_tree(_read(path), path.parent)
