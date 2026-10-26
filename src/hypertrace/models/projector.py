"""Learnable map from the frozen VLM space into the concept space.

image latent (frozen) --linear/MLP--> tangent vector --scale, clip--> to_space() --> point
text  latent (frozen) --linear-----> tangent vector --scale, clip--> to_space() --> point
                                     (only for concepts.mode=text; free concept points skip it)

* CT-CLIP latents are L2-normalised (per-dim magnitude ~1/sqrt(d_in)), so inputs
  are first rescaled by sqrt(d_in) to unit per-dim scale (``unit_norm_inputs``).
  Features that are not L2-normalised (token_mean, lesion-encoder embeddings) are
  z-scored on the training split instead (``data.standardize``). Without this every
  point starts squeezed at the origin and concept logits are nearly constant.
* Learnable scales alpha (init 1/sqrt(d)) then place points at moderate radius,
  where the Poincaré ball is numerically well behaved (as in MERU).
* Optional feature clipping (Guo et al., CVPR 2022, "Clipped hyperbolic
  classifiers") caps the tangent norm so points cannot reach the boundary,
  where gradients vanish.
* The Euclidean twin uses the exact same projector; only ``to_space`` differs. The cap
  ``clip_r`` is set per geometry (``resolve_clip_r``): the Euclidean space has no boundary,
  and a cap tuned for the ball pinned every Euclidean lesion to a ring at d = 2.
* ``input_dropout`` drops input features before the projection (regulariser for
  4,096-d lesion embeddings and a few hundred training nodules).
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from hypertrace.geometry import ConceptGeometry


def resolve_clip_r(clip_r, geometry_name: str) -> float | None:
    """``clip_r`` may be one number (every geometry) or a mapping {geometry name: number | null}."""
    if isinstance(clip_r, dict):
        if geometry_name not in clip_r:
            raise ValueError(f"model.projector.clip_r has no entry for geometry {geometry_name!r} "
                             f"(entries: {sorted(clip_r)})")
        clip_r = clip_r[geometry_name]
    return None if clip_r is None else float(clip_r)


def _mlp(in_dim: int, out_dim: int, hidden_dim: int | None, dropout: float) -> nn.Module:
    if not hidden_dim:
        return nn.Linear(in_dim, out_dim)
    return nn.Sequential(
        nn.Linear(in_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim, out_dim)
    )


class ConceptSpaceProjector(nn.Module):
    def __init__(
        self,
        geometry: ConceptGeometry,
        image_dim: int,
        text_dim: int | None,
        embed_dim: int = 64,
        image_hidden_dim: int | None = None,
        dropout: float = 0.0,
        share_projection: bool = False,
        clip_r: float | dict | None = 2.0,
        unit_norm_inputs: bool = True,
        input_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.geometry = geometry
        self.clip_r = resolve_clip_r(clip_r, geometry.name)
        self.input_dropout = nn.Dropout(input_dropout) if input_dropout else nn.Identity()
        self.image_in_scale = math.sqrt(image_dim) if unit_norm_inputs else 1.0
        self.text_in_scale = math.sqrt(text_dim) if (unit_norm_inputs and text_dim) else 1.0
        self.image_proj = _mlp(image_dim, embed_dim, image_hidden_dim, dropout)
        init = math.log(1.0 / math.sqrt(embed_dim))
        if text_dim is None:  # free concept points: no text map
            self.log_alpha_image = nn.Parameter(torch.tensor(init))
            return
        if share_projection:
            if image_dim != text_dim or image_hidden_dim:
                raise ValueError(
                    "share_projection needs image_dim == text_dim and a linear image projection "
                    "(true for image_feature=masked_latent)."
                )
            self.text_proj = self.image_proj
        else:
            self.text_proj = nn.Linear(text_dim, embed_dim)
        # registration order kept as in v1 (parameter order feeds the optimiser groups)
        self.log_alpha_image = nn.Parameter(torch.tensor(init))
        self.log_alpha_text = nn.Parameter(torch.tensor(init))

    def _clip(self, v: Tensor) -> Tensor:
        if self.clip_r is None:
            return v
        norm = v.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        return v * torch.clamp(self.clip_r / norm, max=1.0)

    def embed_image(self, x: Tensor) -> Tensor:
        v = self.image_proj(self.input_dropout(x) * self.image_in_scale) * self.log_alpha_image.exp()
        return self.geometry.to_space(self._clip(v))

    def embed_text(self, t: Tensor) -> Tensor:
        v = self.text_proj(t * self.text_in_scale) * self.log_alpha_text.exp()
        return self.geometry.to_space(self._clip(v))
