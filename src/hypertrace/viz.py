"""Geometry-aware 2-D views of a trained concept space.

Two layouts, both computed from the run's own geometry:

``radial``  radius = the point's exact distance from the origin, i.e. its *depth*
            (hyperbolic distance d(0, x) for Poincaré, Euclidean norm for the twin).
            Angle = a 2-D PCA of the point's direction in the tangent space at the
            origin. Depth is exact; only angles are approximate. This is the honest
            view of the hierarchy claim (general concepts central, specific outward).
``disk``    curved spaces: tangent vectors (log_0) -> 2-D PCA -> exp_0 into the 2-D space of the
            same curvature: a Poincaré disk (κ < 0, boundary dashed) or the stereographic view of
            a sphere (κ > 0, equator dashed; the far hemisphere lies outside it). Distortion grows
            away from the origin. Euclidean: plain 2-D PCA of the points.

Product runs (one factor per concept group) give one view per factor, in that factor's geometry.

Drawn edges are straight chords between parent and child. In the disk the true
geodesics are arcs; chords are an approximation for readability.

The model is rebuilt from the run's ``config.yaml`` + ``checkpoints/best.pt`` and the
embedding cache, so every nodule (train/val/test) can be shown, not only the test fold.

Concept groups (``concept_groups``): the trunk is every node that is an ancestor of ALL
scored concepts (chest -> lung -> pulmonary lesion -> nodule); a node's group is the first
node below the trunk on its path (size, shape, margin, texture, calcification). Figures
colour concepts by group.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from hypertrace.concepts import ConceptHierarchy
from hypertrace.data.dataset import ConceptData
from hypertrace.geometry import curvature_of
from hypertrace.geometry import kstereo as ks
from hypertrace.models.model import ConceptSpaceModel, build_model


@dataclass
class SpaceView:
    run_name: str
    geometry: str
    curvature: float | None
    node_names: list[str]
    node_parents: list[str | None]
    node_scored: list[bool]
    node_tree_depth: np.ndarray        # (N,) depth in the concept tree
    node_depth: np.ndarray             # (N,) geometric depth (hyperbolic or Euclidean)
    node_group: list[str | None]       # concept group of each node (None: trunk node)
    lesions: pd.DataFrame              # nodule_id, patient_id, split, depth, soft_*, malignancy_mean
    radial_nodes: np.ndarray           # (N, 2)
    radial_lesions: np.ndarray         # (M, 2)
    disk_nodes: np.ndarray             # (N, 2)
    disk_lesions: np.ndarray           # (M, 2)
    disk_radius: float | None          # boundary radius for the Poincaré disk, None for Euclidean
    node_display: list[str | None] | None = None  # clinical display terms (ontology trees), else None
    kappa: float = 0.0                 # signed curvature (-c for Poincaré)
    disk_kind: str | None = None       # ball (boundary at disk_radius) | sphere (equator at disk_radius) | None
    group_order: list[str] | None = None  # every concept group of the tree, so colours match across factor rows


def concept_groups(h: ConceptHierarchy) -> list[str | None]:
    """Group of every node: the first node below the trunk on its root path (None for trunk nodes)."""
    return [h.group_of(n) for n in h.names]


def _pca2(x: np.ndarray, fit_on: np.ndarray) -> np.ndarray:
    """Project x onto the top-2 principal axes of ``fit_on`` (uncentred: the origin is meaningful)."""
    _, _, vt = np.linalg.svd(fit_on, full_matrices=False)
    return x @ vt[:2].T


def load_run_model(run_dir: str | Path, device: str = "cpu") -> tuple[dict, ConceptSpaceModel, ConceptData]:
    run_dir = Path(run_dir)
    cfg = yaml.safe_load((run_dir / "config.yaml").read_text())
    cfg.pop("_meta", None)
    tree = cfg.pop("concepts")
    hierarchy = ConceptHierarchy.from_config(tree)
    data = ConceptData(manifest_path=cfg["paths"]["manifest"], cache_dir=cfg["paths"]["cache_dir"],
                       hierarchy=hierarchy, **cfg["data"])
    model = build_model(cfg, hierarchy, data)
    state = torch.load(run_dir / "checkpoints" / "best.pt", map_location=device, weights_only=False)["model"]
    model.load_state_dict(state)
    return cfg, model.to(device).eval(), data


def _lesion_table(data: ConceptData, depth: np.ndarray) -> pd.DataFrame:
    df = data.df
    split = np.full(len(df), "train", dtype=object)
    split[data.splits["val"]] = "val"
    split[data.splits["test"]] = "test"
    les = pd.DataFrame({"nodule_id": df["nodule_id"], "patient_id": df["patient_id"], "split": split,
                        "depth": depth})
    for k, name in enumerate(data.concept_names):
        les[f"soft_{name}"] = data.soft[:, k].numpy()
    mal = [col for col in df.columns if col.startswith("malignancy_r")]
    les["malignancy_mean"] = df[mal].mean(axis=1, skipna=True).to_numpy() if mal else np.nan
    return les


def _view(run_name: str, geom, nodes: torch.Tensor, lesions: torch.Tensor, node_idx: list[int],
          data: ConceptData) -> SpaceView:
    """Radial and disk layouts of ``nodes`` (the tree nodes ``node_idx``) and every lesion, in ``geom``."""
    kappa = curvature_of(geom)
    t_nodes, t_les = geom.to_tangent(nodes).numpy(), geom.to_tangent(lesions).numpy()
    d_nodes, d_les = geom.depth(nodes).numpy(), geom.depth(lesions).numpy()

    # radial: exact depth as radius, PCA of unit directions as angle
    def unit(v: np.ndarray) -> np.ndarray:
        return v / np.clip(np.linalg.norm(v, axis=1, keepdims=True), 1e-12, None)
    fit = np.concatenate([unit(t_nodes), unit(t_les)])
    radial_n = unit(_pca2(unit(t_nodes), fit)) * d_nodes[:, None]
    radial_l = unit(_pca2(unit(t_les), fit)) * d_les[:, None]

    # disk: tangent PCA -> exp_0 in 2-D with the same curvature (Poincaré disk / stereographic sphere);
    # plain PCA when flat
    fit_t = np.concatenate([t_nodes, t_les])
    p_n, p_l = _pca2(t_nodes, fit_t), _pca2(t_les, fit_t)
    if abs(kappa) > 1e-6:
        disk_n = ks.expmap0(torch.from_numpy(p_n).float(), kappa).numpy()
        disk_l = ks.expmap0(torch.from_numpy(p_l).float(), kappa).numpy()
        radius, kind = 1 / np.sqrt(abs(kappa)), ("ball" if kappa < 0 else "sphere")
    else:
        disk_n, disk_l, radius, kind = p_n, p_l, None, None

    h = data.hierarchy
    keep = set(node_idx)
    sub = [h.nodes[i] for i in node_idx]
    return SpaceView(
        run_name=run_name, geometry=geom.name, curvature=float(geom.c) if geom.name == "poincare" else None,
        node_names=[n.name for n in sub],
        node_parents=[n.parent if n.parent is not None and h.index(n.parent) in keep else None for n in sub],
        node_scored=[n.is_scored for n in sub], node_tree_depth=h.depths[node_idx].numpy(), node_depth=d_nodes,
        node_group=[h.group_of(n.name) for n in sub], lesions=_lesion_table(data, d_les),
        node_display=[n.display for n in sub],
        radial_nodes=radial_n, radial_lesions=radial_l, disk_nodes=disk_n, disk_lesions=disk_l, disk_radius=radius,
        kappa=kappa, disk_kind=kind, group_order=list(h.groups()),
    )


@torch.no_grad()
def build_views(run_dir: str | Path, device: str = "cpu") -> list[SpaceView]:
    """One view per run; a product run gives one view per factor (concept group), in its own geometry."""
    run_dir = Path(run_dir)
    cfg, model, data = load_run_model(run_dir, device)
    lesions = model.projector.embed_image(data.x.to(device)).cpu()
    h = data.hierarchy
    if model.geometry.name != "product":
        return [_view(run_dir.name, model.geometry, model.concept_points().cpu(), lesions, list(range(len(h))),
                      data)]
    views = []
    parts = model.geometry.split(lesions)
    for f, info in enumerate(model.factor_info()):
        g = model.geometry.factors[f]
        views.append(_view(f"{run_dir.name} · {info['group']} ({g.name})", g, model.factor_points[f].detach().cpu(),
                           parts[f], [h.index(n) for n in info["nodes"]], data))
    return views


def build_view(run_dir: str | Path, device: str = "cpu") -> SpaceView:
    """The single view of a one-space run (the first factor of a product run)."""
    return build_views(run_dir, device)[0]
