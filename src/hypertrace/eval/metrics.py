"""Concept and hierarchy metrics (proposal Sec. 7B and 7C).

Computed with numpy/sklearn on the full split at epoch end: the data are small
(hundreds of nodules per split) and per-concept NaN masks are easier to get
right here than in streaming metrics. Training on cached embeddings runs on one
device, so no cross-process gathering is needed.
"""

from __future__ import annotations

import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

from hypertrace.concepts import ConceptHierarchy
from hypertrace.geometry import ConceptGeometry


def concept_metrics(
    probs: np.ndarray, soft: np.ndarray, names: list[str], threshold: float = 0.5
) -> dict[str, float]:
    """probs, soft: (N, K). Binary view = soft >= threshold. Concepts with one class are skipped for AUROC/AP."""
    out: dict[str, float] = {}
    aurocs, aps, f1s, briers = [], [], [], []
    for k, name in enumerate(names):
        m = ~np.isnan(soft[:, k])
        if m.sum() == 0:
            continue
        y_soft, p = soft[m, k], probs[m, k]
        y = (y_soft >= threshold).astype(int)
        brier = float(np.mean((p - y_soft) ** 2))
        out[f"brier/{name}"] = brier
        briers.append(brier)
        f1 = float(f1_score(y, (p >= 0.5).astype(int), zero_division=0))
        out[f"f1/{name}"] = f1
        f1s.append(f1)
        if 0 < y.sum() < len(y):
            a = float(roc_auc_score(y, p))
            ap = float(average_precision_score(y, p))
            out[f"auroc/{name}"], out[f"auprc/{name}"] = a, ap
            aurocs.append(a)
            aps.append(ap)
    out["auroc_macro"] = float(np.mean(aurocs)) if aurocs else float("nan")
    out["auprc_macro"] = float(np.mean(aps)) if aps else float("nan")
    out["f1_macro"] = float(np.mean(f1s)) if f1s else float("nan")
    out["brier_macro"] = float(np.mean(briers)) if briers else float("nan")
    return out


def consistency_metrics(probs: np.ndarray, hierarchy: ConceptHierarchy, threshold: float = 0.5) -> dict[str, float]:
    """Are the predicted concepts logically coherent with the tree? (lower is better)

    parent_child_violation  fraction of (nodule, scored parent->child pair) with the child predicted
                            present but the parent absent (e.g. part-solid without subsolid)
    exclusive_violation     fraction of (nodule, exclusive group) without exactly one category
                            predicted present (size bins, shape, nodule type, calcification)
    """
    pred = probs >= threshold
    out: dict[str, float] = {}
    pairs = hierarchy.scored_edges()
    if pairs:
        out["parent_child_violation"] = float(np.mean([(pred[:, c] & ~pred[:, p]).mean() for p, c in pairs]))
    groups = hierarchy.exclusive_groups()
    if groups:
        out["exclusive_violation"] = float(np.mean([(pred[:, g].sum(1) != 1).mean() for g in groups]))
    return out


def tree_map(dist: np.ndarray, adjacency: np.ndarray) -> float:
    """Mean average precision of reconstructing each node's tree neighbours from distances
    (Nickel & Kiela, 2017): for node u, rank the others by distance; AP over its neighbours."""
    aps = []
    for u in range(len(dist)):
        nbrs = np.flatnonzero(adjacency[u])
        if len(nbrs) == 0:
            continue
        d = np.delete(dist[u], u)
        is_nbr = np.delete(adjacency[u], u).astype(bool)
        precs = []
        for v in np.flatnonzero(is_nbr):
            ranked = d <= d[v]
            precs.append(is_nbr[ranked].sum() / ranked.sum())
        aps.append(np.mean(precs))
    return float(np.mean(aps))


@torch.no_grad()
def hierarchy_metrics(
    geometry: ConceptGeometry, nodes: torch.Tensor, edges: torch.Tensor, non_edges: torch.Tensor,
    tree_depths: torch.Tensor, tree_paths: torch.Tensor | None = None,
) -> dict[str, float]:
    """How well the concept embeddings encode the tree.

    edge_in_cone         fraction of parent->child edges with the child inside the parent's cone (higher is better)
    non_edge_in_cone     fraction of unrelated pairs wrongly inside a cone (lower is better)
    depth_spearman       rank correlation between tree depth and distance from origin
                         (positive = general concepts near the centre, specific ones outward)
    tree_map             mean average precision of recovering tree neighbours from distances (1 = perfect)
    dist_spearman        rank correlation between embedded distance and tree path length over all pairs
    """
    pos = geometry.entailment_violation(nodes[edges[:, 0]], nodes[edges[:, 1]])
    neg = geometry.entailment_violation(nodes[non_edges[:, 0]], nodes[non_edges[:, 1]])
    rho = spearmanr(tree_depths.cpu().numpy(), geometry.depth(nodes).cpu().numpy()).statistic
    out = {
        "edge_in_cone": float((pos <= 1e-6).float().mean()),
        "non_edge_in_cone": float((neg <= 1e-6).float().mean()),
        "depth_spearman": float(rho),
    }
    if tree_paths is not None:
        d = geometry.pairwise_dist(nodes, nodes).cpu().numpy()
        paths = tree_paths.cpu().numpy()
        iu = np.triu_indices(len(paths), k=1)
        out["tree_map"] = tree_map(d, paths == 1)
        out["dist_spearman"] = float(spearmanr(paths[iu], d[iu]).statistic)
    return out


def _auroc(y: np.ndarray, s: np.ndarray) -> float:
    m = ~np.isnan(y)
    if len(np.unique(y[m])) < 2:
        return float("nan")
    return float(roc_auc_score(y[m], s[m]))


def group_metrics(model, probs: np.ndarray, soft: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    """Per concept group (size, nodule type, shape, margin, calcification, ...), keys ``group/<g>/<metric>``.

    Each metric matches the structure the group is supposed to have, so the benefit of a geometry can be read
    on the property it should help with. Works for single-space and product models through
    ``model.group_view(g)`` -> (geometry, points, node indices) of the space the group's concepts live in.

    auroc                  mean AUROC of the group's scored concepts (binary view, soft >= threshold)
    exclusive_violation    exclusive nodes in the group with >= 2 scored children: fraction of nodules
                           without exactly one predicted category (lower is better)
    parent_child_violation scored parent -> child pairs inside the group (child present, parent absent)
    ordinal_spearman       ordinal nodes (size bins): Spearman between the true bin (argmax of soft labels)
                           and the predicted expected bin sum_i i p_i / sum_i p_i
    ordinal_dist_spearman  ordinal nodes: Spearman between the distance of two bins' concept points and
                           how many bins apart they are (1 = bins laid out in order)
    edge_in_cone           group subtree edges with the child inside its parent's cone
    depth_spearman         tree depth vs distance from the origin, inside the group subtree
    tree_map               tree neighbours recovered from distances, inside the group subtree
    """
    h = model.hierarchy
    col = {n: i for i, n in enumerate(h.scored_names)}
    y_all = np.where(np.isnan(soft), np.nan, (soft >= threshold).astype(float))
    pred = probs >= threshold
    out: dict[str, float] = {}
    for g, members in h.groups().items():
        scored = [n for n in members if n in col]
        if not scored:
            continue
        key = f"group/{g}/"
        aucs = [v for v in (_auroc(y_all[:, col[n]], probs[:, col[n]]) for n in scored) if v == v]
        out[key + "auroc"] = float(np.mean(aucs)) if aucs else float("nan")
        excl, ords, ord_geo = [], [], []
        geom, pts, idx = model.group_view(g)
        local = {h.names[i]: j for j, i in enumerate(idx)}
        for n in members:
            node = h.nodes[h.index(n)]
            ch = [c for c in h.children(n) if c in col]
            if node.exclusive and len(ch) >= 2:
                excl.append(float((pred[:, [col[c] for c in ch]].sum(1) != 1).mean()))
            if node.ordinal and len(ch) >= 3:
                s = soft[:, [col[c] for c in ch]]
                ok = ~np.isnan(s).any(1)
                p = probs[ok][:, [col[c] for c in ch]]
                expected = (p * np.arange(len(ch))).sum(1) / np.clip(p.sum(1), 1e-9, None)
                ords.append(float(spearmanr(s[ok].argmax(1), expected).statistic))
                if all(c in local for c in ch):
                    d = geom.pairwise_dist(pts[[local[c] for c in ch]], pts[[local[c] for c in ch]]).cpu().numpy()
                    iu = np.triu_indices(len(ch), k=1)
                    ord_geo.append(float(spearmanr(d[iu], (iu[1] - iu[0])).statistic))
        if excl:
            out[key + "exclusive_violation"] = float(np.mean(excl))
        if ords:
            out[key + "ordinal_spearman"] = float(np.mean(ords))
        if ord_geo:
            out[key + "ordinal_dist_spearman"] = float(np.mean(ord_geo))
        pc = [(col[a], col[n]) for n in scored for a in h.ancestors(n) if a in col and a in members]
        if pc:
            out[key + "parent_child_violation"] = float(np.mean([(pred[:, c] & ~pred[:, p]).mean() for p, c in pc]))
        edges = [(local[h.nodes[i].parent], j) for j, i in enumerate(idx) if h.nodes[i].parent in local]
        if edges:
            e = torch.tensor(edges, dtype=torch.long)
            viol = geom.entailment_violation(pts[e[:, 0]], pts[e[:, 1]])
            out[key + "edge_in_cone"] = float((viol <= 1e-6).float().mean())
            depths = h.depths[idx].cpu().numpy()
            if len(np.unique(depths)) > 1:
                out[key + "depth_spearman"] = float(spearmanr(depths, geom.depth(pts).cpu().numpy()).statistic)
            adj = np.zeros((len(idx), len(idx)), bool)
            for a, b in edges:
                adj[a, b] = adj[b, a] = True
            out[key + "tree_map"] = tree_map(geom.pairwise_dist(pts, pts).cpu().numpy(), adj)
    return out
