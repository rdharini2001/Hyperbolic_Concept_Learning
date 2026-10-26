"""Read concept scores off the concept space.

logit_k = -d(lesion, concept_k) / tau + b_k ,  score_k = sigmoid(logit_k)

The score of a concept is a monotone function of how close the lesion sits to
that concept's text anchor, so the bottleneck is literally a readout of the
geometry (no separate classifier). Independent sigmoids, because LIDC concepts
are not mutually exclusive. ``per_concept_tau`` gives each concept its own temperature.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from hypertrace.geometry import ConceptGeometry


class DistanceConceptHead(nn.Module):
    def __init__(self, geometry: ConceptGeometry, n_concepts: int, init_tau: float = 1.0,
                 learn_tau: bool = True, per_concept_tau: bool = False) -> None:
        super().__init__()
        self.geometry = geometry
        log_tau = torch.tensor(float(init_tau)).log()
        if per_concept_tau:  # one temperature per concept: rare concepts may need a different scale
            log_tau = log_tau.repeat(n_concepts)
        if learn_tau:
            self.log_tau = nn.Parameter(log_tau)
        else:
            self.register_buffer("log_tau", log_tau)
        self.bias = nn.Parameter(torch.zeros(n_concepts))

    def forward(self, lesion: Tensor, concepts: Tensor) -> Tensor:
        """(B, D) lesion points, (K, D) concept points -> (B, K) logits."""
        d = self.geometry.pairwise_dist(lesion, concepts)
        return -d / self.log_tau.exp().clamp_min(1e-3) + self.bias
