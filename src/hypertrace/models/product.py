"""Product concept space: one factor (and one geometry) per concept group.

    lesion embedding --linear--> tangent vector of size D = d_1 + ... + d_F, split per factor
       factor f: clip, then exp_0 of that factor's geometry -> point x_f in M_f
    concept k of group g lives in the factor of g only:
       logit_k = -d_{M_f}(x_f, p_k) / tau + b_k

Each concept group (size, nodule type, shape, margin, calcification: the children of the trunk
chest -> ... -> nodule, see ``ConceptHierarchy.groups``) gets its own factor, so a group's
concepts are read from coordinates no other group uses, in the geometry assigned to that group.
The assignment is a named table in the config (``geometry.product.assignments``): matched
(hierarchy -> Poincaré, flat exclusive categories -> sphere, ordered bins and independent ratings
-> Euclidean), swapped (deliberately mismatched), uniform (one geometry everywhere) and learned
(κ-stereographic with a learned curvature per factor, starting flat). All of them split the same
total dimension D (``model.projector.embed_dim``) equally across factors, so they compare with each
other and with a single space of dimension D.

Trunk nodes (chest ... nodule) hold for every input nodule and are not embedded. Groups without
any scored concept are skipped unless ``embed_unscored_groups`` is set (their lesion coordinates
would have no supervision). Within each factor the hierarchy loss and the lesion-entailment loss
use that factor's geometry and the edges of that group's subtree. On the sphere these cones are only
formal, so sphere factors skip both losses unless ``cones_on_sphere`` is set. Concept points are
updated with Riemannian Adam in their own factor's geometry (``optim.py``, ``geometry_index``).
"""

from __future__ import annotations

import logging
from typing import Any

import torch
from torch import Tensor, nn

from hypertrace.concepts import ConceptHierarchy
from hypertrace.eval.metrics import hierarchy_metrics
from hypertrace.geometry import ConceptGeometry, build_geometry, curvature_of
from hypertrace.models.losses import concept_bce, hierarchy_entailment, image_entailment
from hypertrace.models.model import CONCEPT_DEFAULTS, init_concept_tangents, split_curvature
from hypertrace.models.projector import ConceptSpaceProjector, resolve_clip_r

log = logging.getLogger(__name__)


def _clip(v: Tensor, r: float | None) -> Tensor:
    if r is None:
        return v
    n = v.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    return v * torch.clamp(r / n, max=1.0)


class ProductGeometry(nn.Module):
    """M_1 x ... x M_F with the product (l2) metric, for whole-lesion quantities (depth, probes)."""

    name = "product"

    def __init__(self, factors: list[ConceptGeometry], dims: list[int], clips: list[float | None],
                 names: list[str]) -> None:
        super().__init__()
        self.factors = nn.ModuleList(factors)
        self.dims, self.clips, self.names = list(dims), list(clips), list(names)
        self.offsets = [sum(self.dims[:i]) for i in range(len(self.dims))]

    def split(self, x: Tensor) -> list[Tensor]:
        return list(torch.split(x, self.dims, dim=-1))

    def to_space(self, v: Tensor) -> Tensor:
        return torch.cat([g.to_space(_clip(u, c)) for g, u, c in zip(self.factors, self.split(v), self.clips,
                                                                         strict=True)], -1)

    def to_tangent(self, x: Tensor) -> Tensor:
        return torch.cat([g.to_tangent(p) for g, p in zip(self.factors, self.split(x), strict=True)], -1)

    def depth(self, x: Tensor) -> Tensor:
        return torch.stack([g.depth(p) for g, p in zip(self.factors, self.split(x), strict=True)], -1).norm(dim=-1)

    def pairwise_dist(self, x: Tensor, y: Tensor) -> Tensor:
        d2 = sum(g.pairwise_dist(a, b) ** 2 for g, a, b in zip(self.factors, self.split(x), self.split(y),
                                                                 strict=True))
        return (d2 + 1e-12).sqrt()


def resolve_assignment(geometry_cfg: dict, hierarchy: ConceptHierarchy) -> list[tuple[str, str]]:
    """[(group, geometry name)] for every embedded group, in tree order."""
    p = geometry_cfg["product"]
    name = p["assignment"]
    tables = p.get("assignments") or {}
    if name not in tables:
        raise ValueError(f"geometry.product.assignment {name!r} is not defined; options: {sorted(tables)}")
    table = dict(tables[name])
    scored = set(hierarchy.scored_names)
    out = []
    for g, members in hierarchy.groups().items():
        has_scored = any(m in scored for m in members)
        if not has_scored and not p.get("embed_unscored_groups", False):
            continue
        geom = table.get(g, table.get("*"))
        if geom is None:
            raise ValueError(f"assignment {name!r} gives no geometry for concept group {g!r} (add it or a '*' entry)")
        out.append((g, geom))
    unknown = sorted(set(table) - set(hierarchy.groups()) - {"*"})
    if unknown:
        raise ValueError(f"assignment {name!r} names groups that are not in the concept tree: {unknown}")
    return out


def split_dims(total: int, n: int, min_dim: int = 1) -> list[int]:
    if total < n * min_dim:
        raise ValueError(f"model.projector.embed_dim = {total} is too small for {n} factors of at least {min_dim} "
                         f"dimension(s) each (product geometry: embed_dim is the TOTAL over factors)")
    base, rest = divmod(total, n)
    return [base + (1 if i < rest else 0) for i in range(n)]


class ProductConceptModel(nn.Module):
    def __init__(self, hierarchy: ConceptHierarchy, image_dim: int, geometry: dict[str, Any],
                 projector: dict[str, Any], head: dict[str, Any], loss: dict[str, Any],
                 concepts: dict[str, Any] | None = None, pos_weight: Tensor | None = None) -> None:
        super().__init__()
        self.hierarchy = hierarchy
        self.loss_cfg = dict(loss)
        self.concept_cfg = {**CONCEPT_DEFAULTS, **(concepts or {})}
        if self.concept_cfg["mode"] != "free" or self.concept_cfg["param"] != "riemannian":
            raise ValueError("the product geometry needs model.concepts.mode=free and param=riemannian")
        pcfg = geometry["product"]
        self.assignment_name = pcfg["assignment"]
        assign = resolve_assignment(geometry, hierarchy)
        dims = split_dims(int(projector.get("embed_dim", 64)), len(assign), int(pcfg.get("min_factor_dim", 1)))
        factors, clips = [], []
        for (_, gname), _d in zip(assign, dims, strict=True):
            factors.append(build_geometry(gname, **geometry.get(gname, {})))
            clips.append(resolve_clip_r(projector.get("clip_r"), gname))
        self.group_names = [g for g, _ in assign]
        self.geometry = ProductGeometry(factors, dims, clips, self.group_names)
        cones_on_sphere = bool(pcfg.get("cones_on_sphere", False))
        self.use_cones = [not (g.name == "sphere" and not cones_on_sphere) for g in factors]
        log.info("Product geometry (%s): %s", self.assignment_name,
                 ", ".join(f"{n}={g.name}[{d}]" for n, g, d in zip(self.group_names, factors, dims, strict=True)))

        proj = {k: v for k, v in projector.items() if k not in ("share_projection", "clip_r")}
        self.projector = ConceptSpaceProjector(self.geometry, image_dim=image_dim, text_dim=None, clip_r=None, **proj)

        groups = hierarchy.groups()
        col = {n: i for i, n in enumerate(hierarchy.scored_names)}
        all_edges = hierarchy.edges(transitive=loss.get("transitive_edges", True)).tolist()
        direct = hierarchy.edges(transitive=False).tolist()
        non_edges = hierarchy.non_edges().tolist()
        self.factor_points = nn.ParameterList()
        self._node_idx, self._scored_local, self._scored_cols = [], [], []
        self._edges, self._direct, self._non_edges = [], [], []
        embedded_scored = []
        for f, (g, gname) in enumerate(assign):
            names = groups[g]
            idx = [hierarchy.index(n) for n in names]
            local = {gi: li for li, gi in enumerate(idx)}
            v = init_concept_tangents(hierarchy, dims[f], self.concept_cfg["init"], None,
                                      float(self.concept_cfg["init_spread"]),
                                      float(self.concept_cfg["init_max_radius"]))[idx]
            with torch.no_grad():
                self.factor_points.append(nn.Parameter(factors[f].to_space(_clip(v, clips[f]))))
            sc = [n for n in names if n in col]
            embedded_scored += sc
            self._node_idx.append(idx)
            self._scored_local.append([names.index(n) for n in sc])
            self._scored_cols.append([col[n] for n in sc])

            def keep(pairs: list[list[int]], local: dict = local) -> Tensor:
                kept = [[local[a], local[b]] for a, b in pairs if a in local and b in local]
                return torch.tensor(kept, dtype=torch.long).reshape(-1, 2)

            self._edges.append(keep(all_edges))
            self._direct.append(keep(direct))
            self._non_edges.append(keep(non_edges))
            log.info("  factor %s: %d nodes (%d scored), %d edges, geometry %s, cones %s", g, len(idx), len(sc),
                     len(self._direct[-1]), gname, "on" if self.use_cones[f] else "off")
        missing = sorted(set(hierarchy.scored_names) - set(embedded_scored))
        if missing:
            raise ValueError(f"scored concepts outside every product factor: {missing}")

        k = len(hierarchy.scored_names)
        log_tau = torch.tensor(float(head.get("init_tau", 0.2))).log()
        if head.get("per_concept_tau", False):
            log_tau = log_tau.repeat(k)
        if head.get("learn_tau", True):
            self.log_tau = nn.Parameter(log_tau)
        else:
            self.register_buffer("log_tau", log_tau)
        self.bias = nn.Parameter(torch.zeros(k))
        self.register_buffer("pos_weight", pos_weight.float() if pos_weight is not None else None, persistent=False)
        self.register_buffer("tree_depths", hierarchy.depths, persistent=False)
        self.register_buffer("tree_paths", hierarchy.path_lengths(), persistent=False)

    # ------------------------------------------------------------------ forward
    def factor(self, group: str) -> int:
        return self.group_names.index(group)

    def concept_points(self) -> Tensor:
        """(N_nodes, D) positions: a node's own factor block filled, every other block at that factor's origin.
        Trunk and skipped-group nodes sit at the origin of every factor."""
        n, D = len(self.hierarchy), sum(self.geometry.dims)
        ref = self.factor_points[0]
        out = ref.new_zeros(n, D)
        for f, pts in enumerate(self.factor_points):
            o, d = self.geometry.offsets[f], self.geometry.dims[f]
            idx = torch.tensor(self._node_idx[f], device=ref.device)
            out = out.index_put((idx.unsqueeze(1), torch.arange(o, o + d, device=ref.device).unsqueeze(0)), pts)
        return out

    def forward(self, x: Tensor) -> dict[str, Tensor]:
        lesion = self.projector.embed_image(x)
        parts = self.geometry.split(lesion)
        tau = self.log_tau.exp().clamp_min(1e-3)
        cols, logits = [], []
        for f, (g, pts) in enumerate(zip(self.geometry.factors, self.factor_points, strict=True)):
            if not self._scored_cols[f]:
                continue
            d = g.pairwise_dist(parts[f], pts[self._scored_local[f]])
            c = torch.tensor(self._scored_cols[f], device=x.device)
            t = tau[c] if tau.ndim else tau
            logits.append(-d / t + self.bias[c])
            cols += self._scored_cols[f]
        order = torch.argsort(torch.tensor(cols, device=x.device))
        return {"logits": torch.cat(logits, -1)[:, order], "lesion": lesion}

    @torch.no_grad()
    def concept_bottleneck(self, x: Tensor) -> Tensor:
        return torch.sigmoid(self(x)["logits"])

    def loss(self, out: dict[str, Tensor], soft: Tensor) -> dict[str, Tensor]:
        cfg = self.loss_cfg
        parts = {"concept": concept_bce(out["logits"], soft,
                                        self.pos_weight if cfg.get("use_pos_weight", False) else None)}
        lesion = self.geometry.split(out["lesion"])
        hier, img = [], []
        for f, (g, pts) in enumerate(zip(self.geometry.factors, self.factor_points, strict=True)):
            if not self.use_cones[f]:
                continue
            dev = pts.device
            if cfg.get("w_hierarchy", 0) > 0 and len(self._edges[f]):
                hier.append(hierarchy_entailment(g, pts, self._edges[f].to(dev), self._non_edges[f].to(dev),
                                                 neg_margin=cfg.get("neg_margin", 0.1),
                                                 max_neg=cfg.get("max_neg", 512)))
            if cfg.get("w_image_entail", 0) > 0 and self._scored_cols[f]:
                img.append(image_entailment(g, lesion[f], pts[self._scored_local[f]],
                                            soft[:, self._scored_cols[f]], None))
        if hier:
            parts["hierarchy"] = torch.stack(hier).mean()
        if img:
            parts["image_entail"] = torch.stack(img).mean()
        total = parts["concept"]
        total = total + cfg.get("w_hierarchy", 0) * parts.get("hierarchy", 0)
        total = total + cfg.get("w_image_entail", 0) * parts.get("image_entail", 0)
        parts["total"] = total
        return parts

    # ------------------------------------------------------------------ optimisation
    @property
    def riemannian_concepts(self) -> bool:
        return True

    def param_groups(self, weight_decay: float, concept_lr: float | None = None,
                     curvature_lr: float | None = None) -> list[dict]:
        special = {id(p) for p in self.factor_points}
        decay = [p for p in self.parameters() if p.requires_grad and p.ndim >= 2 and id(p) not in special]
        no_decay = [p for p in self.parameters() if p.requires_grad and p.ndim < 2 and id(p) not in special]
        groups = [{"params": decay, "weight_decay": weight_decay}, {"params": no_decay, "weight_decay": 0.0}]
        for f, p in enumerate(self.factor_points):
            g = {"params": [p], "weight_decay": 0.0, "manifold": True, "geometry_index": f}
            if concept_lr is not None:
                g["lr"] = concept_lr
            groups.append(g)
        return split_curvature(self, groups, curvature_lr)

    def manifold_geometries(self) -> list[ConceptGeometry]:
        return list(self.geometry.factors)

    def curvature_log(self) -> dict[str, float]:
        return {f"train/curvature/{n}": curvature_of(g) for n, g in zip(self.group_names, self.geometry.factors,
                                                                          strict=True)}

    # ------------------------------------------------------------------ evaluation
    @torch.no_grad()
    def tree_metrics(self) -> dict[str, float]:
        """Hierarchy metrics inside each factor (``<group>/<metric>``) and their mean over factors (cone
        metrics: mean over the factors that train cones)."""
        out: dict[str, float] = {}
        per: dict[str, list[float]] = {}
        for f, (name, g, pts) in enumerate(zip(self.group_names, self.geometry.factors, self.factor_points,
                                               strict=True)):
            if not len(self._direct[f]):
                continue
            idx = self._node_idx[f]
            dev = pts.device
            m = hierarchy_metrics(g, pts, self._direct[f].to(dev), self._non_edges[f].to(dev),
                                  self.tree_depths[idx], self.tree_paths[idx][:, idx])
            for k, v in m.items():
                out[f"{name}/{k}"] = v
                if k in ("edge_in_cone", "non_edge_in_cone") and not self.use_cones[f]:
                    continue  # cones are not trained in this factor (sphere): keep it out of the mean
                per.setdefault(k, []).append(v)
        for k, vs in per.items():
            finite = [v for v in vs if v == v]
            out[k] = float(sum(finite) / len(finite)) if finite else float("nan")
        return out

    @torch.no_grad()
    def group_view(self, group: str) -> tuple[ConceptGeometry, Tensor, list[int]]:
        f = self.factor(group)
        return self.geometry.factors[f], self.factor_points[f].detach(), list(self._node_idx[f])

    def factor_info(self) -> list[dict]:
        return [{"group": n, "geometry": g.name, "curvature": curvature_of(g), "dim": d, "offset": o,
                 "nodes": [self.hierarchy.names[i] for i in self._node_idx[f]], "cones": self.use_cones[f]}
                for f, (n, g, d, o) in enumerate(zip(self.group_names, self.geometry.factors, self.geometry.dims,
                                                     self.geometry.offsets, strict=True))]
