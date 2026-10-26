"""ConceptSpaceModel: Part 2 of the overview figure as a plain ``nn.Module``.

lesion embedding (cached, frozen encoder)
   -> ConceptSpaceProjector   linear map to a tangent vector at the origin, then expmap_0
                              onto the Poincaré ball (identity for the Euclidean twin)
   -> DistanceConceptHead     concept score = sigmoid(-d_geodesic(lesion, concept) / tau + b)
plus every node of the concept tree embedded in the same ball (entailment-cone losses).

Concept points (``concepts.mode``):

``free``  (default) every node of the tree is its own point in the space. With
          ``param: riemannian`` the parameter IS the point on the ball and is updated with
          Riemannian Adam (``hypertrace/optim.py``); with ``param: tangent`` it is a tangent
          vector at the origin mapped by expmap_0 (ablation: reparameterised, plain AdamW).
          Initialised from the tree (``init: tree``: depth sets the radius, children start
          near their parent's direction) or from the encoder's prompt embeddings
          (``init: text``: principal directions of the centred prompt embeddings).
``text``  (v1) points are the frozen prompt embeddings passed through a learnable linear
          map. Their positions are tied to how similar the prompts are in the text encoder,
          which collapsed all concepts to nearly one point on real CT-CLIP features.

All distances, cones and depths use the closed-form Poincaré formulas in
``geometry/poincare.py`` on points inside the ball. See docs/IMPLEMENTATION.md.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from hypertrace.concepts import ConceptHierarchy
from hypertrace.eval.metrics import hierarchy_metrics
from hypertrace.geometry import ConceptGeometry, build_geometry, curvature_of
from hypertrace.models.concept_head import DistanceConceptHead
from hypertrace.models.losses import concept_bce, hierarchy_entailment, image_entailment
from hypertrace.models.projector import ConceptSpaceProjector

CONCEPT_DEFAULTS = {"mode": "free", "param": "riemannian", "init": "tree", "init_spread": 0.5,
                    "init_max_radius": 2.0}


def init_concept_tangents(hierarchy: ConceptHierarchy, dim: int, init: str = "tree",
                          text_latents: Tensor | None = None, spread: float = 0.5,
                          max_radius: float = 2.0) -> Tensor:
    """Tangent vectors at the origin, one per node: radius grows with tree depth.

    tree: the root gets a random direction; each child starts at its parent's direction plus
          noise of relative size ``spread``, so subtrees start in their own angular sector.
    text: direction = node's coordinates on the top principal axes of the CENTRED prompt
          embeddings (centring removes the component all radiology prompts share).
    Radius in the tangent space: max_radius * (depth + 1) / (max_depth + 2), i.e. general
    concepts near the origin and specific ones further out, as the entailment cones need.
    Uses the global torch RNG (seeded per run).
    """
    depth = hierarchy.depths
    radius = max_radius * (depth + 1) / (depth.max() + 2)
    n = len(hierarchy)
    if init == "tree":
        dirs = torch.zeros(n, dim)
        for i in sorted(range(n), key=lambda k: float(depth[k])):
            parent = hierarchy.nodes[i].parent
            noise = F.normalize(torch.randn(dim), dim=0)
            if parent is None:
                dirs[i] = noise
            else:
                dirs[i] = F.normalize(dirs[hierarchy.index(parent)] + spread * noise, dim=0)
    elif init == "text":
        if text_latents is None:
            raise ValueError("concepts.init=text needs prompt embeddings (text.pt in the cache); "
                             "this encoder has none. Use concepts.init=tree.")
        t = text_latents.float() - text_latents.float().mean(0, keepdim=True)
        _, _, vt = torch.linalg.svd(t, full_matrices=False)
        coords = t @ vt[: min(dim, vt.shape[0])].T
        coords = F.pad(coords, (0, dim - coords.shape[1]))
        coords = coords + 1e-3 * torch.randn_like(coords)  # break ties for identical prompts
        dirs = F.normalize(coords, dim=-1)
    else:
        raise ValueError(f"concepts.init must be 'tree' or 'text', got {init!r}")
    return dirs * radius.unsqueeze(1)


def split_curvature(module, groups: list[dict], curvature_lr: float | None) -> list[dict]:
    """Move learned curvatures (``log_c``, ``kappa_raw``) out of the no-decay group into their own group
    with ``curvature_lr`` (only when it is set, so default runs keep the original grouping)."""
    if curvature_lr is None:
        return groups
    curv = [p for n, p in module.named_parameters() if p.requires_grad and n.endswith(("log_c", "kappa_raw"))]
    if not curv:
        return groups
    ids = {id(p) for p in curv}
    for g in groups:
        g["params"] = [p for p in g["params"] if id(p) not in ids]
    return [*groups, {"params": curv, "weight_decay": 0.0, "lr": curvature_lr}]


class ConceptSpaceModel(nn.Module):
    def __init__(
        self,
        hierarchy: ConceptHierarchy,
        image_dim: int,
        geometry: dict[str, Any],
        projector: dict[str, Any],
        head: dict[str, Any],
        loss: dict[str, Any],
        concepts: dict[str, Any] | None = None,
        text_latents: Tensor | None = None,
        pos_weight: Tensor | None = None,
    ) -> None:
        super().__init__()
        self.hierarchy = hierarchy
        self.loss_cfg = dict(loss)
        self.concept_cfg = {**CONCEPT_DEFAULTS, **(concepts or {})}
        mode, param = self.concept_cfg["mode"], self.concept_cfg["param"]
        self.geometry = build_geometry(geometry["name"], **geometry.get(geometry["name"], {}))

        if mode == "text":
            if text_latents is None:
                raise ValueError("concepts.mode=text needs prompt embeddings (text.pt); use concepts.mode=free.")
            self.projector = ConceptSpaceProjector(self.geometry, image_dim=image_dim,
                                                   text_dim=text_latents.shape[1], **projector)
            self.register_buffer("text_latents", text_latents.float())
        elif mode == "free":
            proj = {k: v for k, v in projector.items() if k != "share_projection"}
            if projector.get("share_projection"):
                raise ValueError("share_projection needs concepts.mode=text (free concepts have no text map).")
            self.projector = ConceptSpaceProjector(self.geometry, image_dim=image_dim, text_dim=None, **proj)
            v = init_concept_tangents(hierarchy, projector.get("embed_dim", 64), self.concept_cfg["init"],
                                      text_latents, float(self.concept_cfg["init_spread"]),
                                      float(self.concept_cfg["init_max_radius"]))
            if param == "riemannian":
                with torch.no_grad():
                    self.concept_param = nn.Parameter(self.geometry.to_space(v))  # a point in the space
            elif param == "tangent":
                self.concept_param = nn.Parameter(v)  # a tangent vector at the origin
            else:
                raise ValueError(f"concepts.param must be 'riemannian' or 'tangent', got {param!r}")
        else:
            raise ValueError(f"concepts.mode must be 'free' or 'text', got {mode!r}")

        self.head = DistanceConceptHead(self.geometry, len(hierarchy.scored_names), **head)
        self.register_buffer("scored_idx", hierarchy.scored_indices, persistent=False)
        self.register_buffer("edges", hierarchy.edges(transitive=loss.get("transitive_edges", True)),
                             persistent=False)
        self.register_buffer("direct_edges", hierarchy.edges(transitive=False), persistent=False)
        self.register_buffer("non_edges", hierarchy.non_edges(), persistent=False)
        self.register_buffer("tree_depths", hierarchy.depths, persistent=False)
        self.register_buffer("tree_paths", hierarchy.path_lengths(), persistent=False)
        always = [hierarchy.index(n) for n in loss.get("image_always_entailed_by", [])]
        self.register_buffer("always_idx", torch.tensor(always, dtype=torch.long), persistent=False)
        self.register_buffer("pos_weight", pos_weight.float() if pos_weight is not None else None,
                             persistent=False)

    @property
    def riemannian_concepts(self) -> bool:
        return self.concept_cfg["mode"] == "free" and self.concept_cfg["param"] == "riemannian"

    def concept_points(self) -> Tensor:
        """(N_nodes, D) positions of every node of the concept tree."""
        if self.concept_cfg["mode"] == "text":
            return self.projector.embed_text(self.text_latents)
        if self.concept_cfg["param"] == "riemannian":
            return self.concept_param
        return self.geometry.to_space(self.concept_param)

    def forward(self, x: Tensor) -> dict[str, Tensor]:
        nodes = self.concept_points()
        lesion = self.projector.embed_image(x)
        logits = self.head(lesion, nodes[self.scored_idx])
        return {"logits": logits, "lesion": lesion, "nodes": nodes}

    @torch.no_grad()
    def concept_bottleneck(self, x: Tensor) -> Tensor:
        """Concept probabilities (B, K), columns ordered as ``hierarchy.scored_names``."""
        return torch.sigmoid(self(x)["logits"])

    def loss(self, out: dict[str, Tensor], soft: Tensor) -> dict[str, Tensor]:
        cfg = self.loss_cfg
        nodes = out["nodes"]
        parts = {"concept": concept_bce(out["logits"], soft,
                                        self.pos_weight if cfg.get("use_pos_weight", False) else None)}
        if cfg.get("w_hierarchy", 0) > 0:
            parts["hierarchy"] = hierarchy_entailment(
                self.geometry, nodes, self.edges, self.non_edges,
                neg_margin=cfg.get("neg_margin", 0.1), max_neg=cfg.get("max_neg", 512),
            )
        if cfg.get("w_image_entail", 0) > 0:
            parts["image_entail"] = image_entailment(
                self.geometry, out["lesion"], nodes[self.scored_idx], soft,
                nodes[self.always_idx] if len(self.always_idx) else None,
            )
        total = parts["concept"]
        total = total + cfg.get("w_hierarchy", 0) * parts.get("hierarchy", 0)
        total = total + cfg.get("w_image_entail", 0) * parts.get("image_entail", 0)
        parts["total"] = total
        return parts

    def param_groups(self, weight_decay: float, concept_lr: float | None = None,
                     curvature_lr: float | None = None) -> list[dict]:
        """Weight decay on matrices only; none on scales, biases, curvature, temperature or concept points.

        Concept points that live on the manifold get their own group flagged ``manifold`` (Riemannian Adam).
        """
        special = {id(self.concept_param)} if hasattr(self, "concept_param") else set()
        decay = [p for p in self.parameters() if p.requires_grad and p.ndim >= 2 and id(p) not in special]
        no_decay = [p for p in self.parameters() if p.requires_grad and p.ndim < 2 and id(p) not in special]
        groups = [{"params": decay, "weight_decay": weight_decay}, {"params": no_decay, "weight_decay": 0.0}]
        if special:
            g = {"params": [self.concept_param], "weight_decay": 0.0, "manifold": self.riemannian_concepts,
                 "geometry_index": 0}
            if concept_lr is not None:
                g["lr"] = concept_lr
            groups.append(g)
        return split_curvature(self, groups, curvature_lr)

    # ---- interface shared with the product model (models/product.py)
    def manifold_geometries(self) -> list[ConceptGeometry]:
        """Geometries addressed by ``geometry_index`` in the optimiser's manifold groups."""
        return [self.geometry]

    def curvature_log(self) -> dict[str, float]:
        """Signed curvature, logged per epoch when it can change (learned curvature)."""
        return {"train/curvature": curvature_of(self.geometry)}

    @torch.no_grad()
    def tree_metrics(self) -> dict[str, float]:
        return hierarchy_metrics(self.geometry, self.concept_points(), self.direct_edges, self.non_edges,
                                 self.tree_depths, self.tree_paths)

    @torch.no_grad()
    def group_view(self, group: str) -> tuple[ConceptGeometry, Tensor, list[int]]:
        """(geometry, points, node indices) of one concept group's subtree, in the space it lives in."""
        idx = [self.hierarchy.index(n) for n in self.hierarchy.groups()[group]]
        return self.geometry, self.concept_points()[idx], idx


def build_model(cfg: dict, hierarchy: ConceptHierarchy, data, pos_weight: Tensor | None = None) -> nn.Module:
    """The model a config describes, for training (train.py) and for loading a run (viz.py)."""
    m = cfg["model"]
    if cfg["geometry"]["name"] == "product":
        from hypertrace.models.product import ProductConceptModel

        return ProductConceptModel(hierarchy, data.image_dim, geometry=cfg["geometry"], projector=m["projector"],
                                   head=m["head"], loss=m["loss"], concepts=m.get("concepts", {}),
                                   pos_weight=pos_weight)
    return ConceptSpaceModel(hierarchy, data.image_dim, geometry=cfg["geometry"], projector=m["projector"],
                             head=m["head"], loss=m["loss"], concepts=m.get("concepts", {"mode": "text"}),
                             text_latents=data.text_latents, pos_weight=pos_weight)
