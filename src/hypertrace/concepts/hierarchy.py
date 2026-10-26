"""Radiology concept tree.

The tree is data, not code: it is defined in ``configs/ontology/*.yaml`` plus a
per-dataset label mapping (``configs/labels/*.yaml``, resolved by ``concepts.ontology``),
or as one self-contained file (``configs/concepts/*.yaml``, v1/v2), and loaded here.
Every node has text prompts (encoded once by the CT-CLIP text encoder).
Only nodes with a ``label`` spec are *scored* concepts, i.e. bottleneck outputs
supervised from LIDC reader ratings. Internal nodes (chest, lung, nodule,
margin, ...) are embedded and shape the hierarchy loss but are not predicted:
for a nodule crop they are always present and would carry no signal.

Siblings are NOT assumed to be mutually exclusive. In LIDC a nodule can be
both lobulated and spiculated, so scored concepts use independent sigmoids.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import torch


@dataclass(frozen=True)
class ConceptNode:
    name: str
    parent: str | None
    prompts: tuple[str, ...]
    label: Mapping[str, Any] | None = None  # spec consumed by concepts.labels
    exclusive: bool = False  # children are mutually exclusive categories (used by consistency metrics)
    display: str | None = None  # clinical term for figures and reports (ontology files)
    radlex: str | None = None  # RadLex RID, when looked up (ontology files)
    ordinal: bool = False  # children are ordered categories (size bins): ordinal metrics

    @property
    def is_scored(self) -> bool:
        return self.label is not None


@dataclass
class ConceptHierarchy:
    name: str
    nodes: list[ConceptNode]
    _index: dict[str, int] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._index = {n.name: i for i, n in enumerate(self.nodes)}
        self._validate()

    # ------------------------------------------------------------------ build
    @classmethod
    def from_config(cls, cfg: Mapping[str, Any]) -> ConceptHierarchy:
        """Build from a dict: ``{name: str, nodes: [ {name, parent, prompts?, label?} ]}``."""
        nodes = []
        for raw in cfg["nodes"]:
            raw = dict(raw)
            prompts = raw.get("prompts") or [raw["name"].replace("_", " ")]
            label = raw.get("label")
            nodes.append(
                ConceptNode(
                    name=str(raw["name"]),
                    parent=raw.get("parent"),
                    prompts=tuple(str(p) for p in prompts),
                    label=dict(label) if label is not None else None,
                    exclusive=bool(raw.get("exclusive", False)),
                    display=raw.get("display"),
                    ordinal=bool(raw.get("ordinal", False)),
                    radlex=(raw.get("radlex") or {}).get("id") if isinstance(raw.get("radlex"), dict) else None,
                )
            )
        return cls(name=str(cfg.get("name", "concepts")), nodes=nodes)

    def _validate(self) -> None:
        if len(self._index) != len(self.nodes):
            raise ValueError("Duplicate concept names in hierarchy")
        roots = [n.name for n in self.nodes if n.parent is None]
        if len(roots) != 1:
            raise ValueError(f"Hierarchy must have exactly one root, found {roots}")
        for n in self.nodes:
            if n.parent is not None and n.parent not in self._index:
                raise ValueError(f"Node '{n.name}' has unknown parent '{n.parent}'")
        for n in self.nodes:  # cycle check: every path must reach the root
            seen, cur = set(), n.name
            while cur is not None:
                if cur in seen:
                    raise ValueError(f"Cycle detected at '{cur}'")
                seen.add(cur)
                cur = self.nodes[self._index[cur]].parent
        if not self.scored_names:
            raise ValueError("No scored concepts: give at least one node a `label` spec")

    # ---------------------------------------------------------------- queries
    def __len__(self) -> int:
        return len(self.nodes)

    def index(self, name: str) -> int:
        return self._index[name]

    @property
    def names(self) -> list[str]:
        return [n.name for n in self.nodes]

    @property
    def root(self) -> str:
        return next(n.name for n in self.nodes if n.parent is None)

    @property
    def scored_names(self) -> list[str]:
        return [n.name for n in self.nodes if n.is_scored]

    @property
    def scored_indices(self) -> torch.Tensor:
        """Positions of scored concepts within ``nodes`` (order defines bottleneck columns)."""
        return torch.tensor([i for i, n in enumerate(self.nodes) if n.is_scored], dtype=torch.long)

    def ancestors(self, name: str) -> list[str]:
        out, cur = [], self.nodes[self._index[name]].parent
        while cur is not None:
            out.append(cur)
            cur = self.nodes[self._index[cur]].parent
        return out

    def depth(self, name: str) -> int:
        return len(self.ancestors(name))

    @property
    def depths(self) -> torch.Tensor:
        return torch.tensor([self.depth(n.name) for n in self.nodes], dtype=torch.float32)

    def edges(self, transitive: bool = False) -> torch.Tensor:
        """(E, 2) tensor of (parent_idx, child_idx).

        ``transitive=True`` adds every ancestor->descendant pair, which gives the
        entailment loss a denser signal (chest must also contain spiculated).
        """
        pairs: list[tuple[int, int]] = []
        for n in self.nodes:
            parents = self.ancestors(n.name) if transitive else ([n.parent] if n.parent else [])
            pairs.extend((self._index[p], self._index[n.name]) for p in parents)
        return torch.tensor(pairs, dtype=torch.long).reshape(-1, 2)

    def non_edges(self) -> torch.Tensor:
        """(E', 2) ordered pairs (a, b) where a is NOT an ancestor of b and a != b.

        Used as negatives for the entailment loss and for hierarchy metrics.
        """
        anc = {n.name: set(self.ancestors(n.name)) for n in self.nodes}
        pairs = [
            (self._index[a.name], self._index[b.name])
            for a in self.nodes
            for b in self.nodes
            if a.name != b.name and a.name not in anc[b.name]
        ]
        return torch.tensor(pairs, dtype=torch.long).reshape(-1, 2)

    def children(self, name: str) -> list[str]:
        return [n.name for n in self.nodes if n.parent == name]

    def exclusive_groups(self) -> list[list[int]]:
        """Scored-column indices of the SCORED children of ``exclusive: true`` nodes with at least two
        scored children. Unscored categories are those the dataset cannot label (e.g. lentiform shape in
        LIDC); the check is "exactly one of the labelled categories", as in v2 where all were labelled."""
        col = {n: i for i, n in enumerate(self.scored_names)}
        groups = []
        for n in self.nodes:
            if n.exclusive:
                ch = [c for c in self.children(n.name) if c in col]
                if len(ch) >= 2:
                    groups.append([col[c] for c in ch])
        return groups

    def trunk(self) -> list[str]:
        """Nodes that are ancestors of (or equal to) EVERY scored concept: chest -> lung -> ... -> nodule.
        For a nodule crop they always hold; product models do not embed them."""
        scored = self.scored_names
        common = set([*self.ancestors(scored[0]), scored[0]])
        for n in scored[1:]:
            common &= set([*self.ancestors(n), n])
        return [n for n in self.names if n in common]

    def group_of(self, name: str) -> str | None:
        """First node below the trunk on the root path of ``name`` (None for trunk nodes)."""
        trunk = set(self.trunk())
        below = [x for x in [*reversed(self.ancestors(name)), name] if x not in trunk]
        return below[0] if below else None

    def groups(self) -> dict[str, list[str]]:
        """{group node: [group node, its descendants]} in tree order, e.g. size -> [size, size_lt6, ...]."""
        out: dict[str, list[str]] = {}
        for n in self.names:
            g = self.group_of(n)
            if g is not None:
                out.setdefault(g, []).append(n)
        return out

    def scored_edges(self) -> list[tuple[int, int]]:
        """(parent column, child column) for scored nodes whose ancestor is also scored."""
        col = {n: i for i, n in enumerate(self.scored_names)}
        return [(col[a], col[n]) for n in self.scored_names for a in self.ancestors(n) if a in col]

    def path_lengths(self) -> torch.Tensor:
        """(N, N) number of tree edges between every pair of nodes."""
        anc = [[n.name, *self.ancestors(n.name)] for n in self.nodes]
        N = len(self.nodes)
        out = torch.zeros(N, N)
        for i in range(N):
            for j in range(N):
                common = next(a for a in anc[i] if a in anc[j])
                out[i, j] = anc[i].index(common) + anc[j].index(common)
        return out

    def all_prompts(self) -> tuple[list[str], torch.Tensor]:
        """Flat prompt list plus owner index, for prompt-ensembled text embeddings."""
        prompts, owner = [], []
        for i, n in enumerate(self.nodes):
            prompts.extend(n.prompts)
            owner.extend([i] * len(n.prompts))
        return prompts, torch.tensor(owner, dtype=torch.long)

    def pretty(self) -> str:
        children: dict[str | None, list[str]] = {}
        for n in self.nodes:
            children.setdefault(n.parent, []).append(n.name)

        lines: list[str] = []

        def walk(name: str, prefix: str) -> None:
            mark = " *" if self.nodes[self._index[name]].is_scored else ""
            lines.append(f"{prefix}{name}{mark}")
            for ch in children.get(name, []):
                walk(ch, prefix + "  ")

        walk(self.root, "")
        return "\n".join(lines) + "\n(* = scored concept)"


def scored_label_specs(h: ConceptHierarchy) -> Sequence[tuple[str, Mapping[str, Any]]]:
    return [(n.name, n.label) for n in h.nodes if n.label is not None]
