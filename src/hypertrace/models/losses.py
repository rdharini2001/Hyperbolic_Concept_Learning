"""Training objectives for the concept space.

L = L_concept + w_h * L_hier + w_img * L_img_entail

L_concept       BCE with soft (reader-mean) targets on scored concepts; NaN targets masked.
L_hier          entailment-cone loss on the concept tree (Ganea et al., 2018):
                positives: child inside parent's cone; negatives: non-ancestor pairs pushed
                at least ``neg_margin`` outside the cone, which stops the trivial solution
                where every concept contains every other.
L_img_entail    lesion should lie inside the cone of concepts it has (weighted by soft
                label) and of always-present ancestors such as ``nodule`` (as in MERU,
                where text entails image). This ties the bottleneck to the hierarchy.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

from hypertrace.geometry import ConceptGeometry


def concept_bce(logits: Tensor, soft: Tensor, pos_weight: Tensor | None = None) -> Tensor:
    mask = ~torch.isnan(soft)
    if not mask.any():
        return logits.sum() * 0.0
    target = torch.nan_to_num(soft, nan=0.0)
    loss = F.binary_cross_entropy_with_logits(logits, target, pos_weight=pos_weight, reduction="none")
    return (loss * mask).sum() / mask.sum()


def hierarchy_entailment(
    geometry: ConceptGeometry, nodes: Tensor, edges: Tensor, non_edges: Tensor | None = None,
    neg_margin: float = 0.1, max_neg: int | None = 512,
) -> Tensor:
    """nodes (N, D); edges/non_edges (E, 2) of (parent, child) indices."""
    pos = geometry.entailment_violation(nodes[edges[:, 0]], nodes[edges[:, 1]]).mean()
    if non_edges is None or len(non_edges) == 0:
        return pos
    if max_neg is not None and len(non_edges) > max_neg:
        non_edges = non_edges[torch.randperm(len(non_edges), device=non_edges.device)[:max_neg]]
    viol = geometry.entailment_violation(nodes[non_edges[:, 0]], nodes[non_edges[:, 1]])
    neg = (neg_margin - viol).clamp_min(0).mean()
    return pos + neg


def image_entailment(
    geometry: ConceptGeometry, lesion: Tensor, scored_nodes: Tensor, soft: Tensor,
    always_nodes: Tensor | None = None,
) -> Tensor:
    """lesion (B, D); scored_nodes (K, D); soft (B, K) with NaN; always_nodes (A, D)."""
    B, K = soft.shape
    viol = geometry.entailment_violation(
        scored_nodes.unsqueeze(0).expand(B, -1, -1), lesion.unsqueeze(1).expand(-1, K, -1)
    )  # (B, K)
    w = torch.nan_to_num(soft, nan=0.0)
    loss = (viol * w).sum() / w.sum().clamp_min(1e-6)
    if always_nodes is not None and len(always_nodes):
        A = always_nodes.shape[0]
        loss = loss + geometry.entailment_violation(
            always_nodes.unsqueeze(0).expand(B, -1, -1), lesion.unsqueeze(1).expand(-1, A, -1)
        ).mean()
    return loss
