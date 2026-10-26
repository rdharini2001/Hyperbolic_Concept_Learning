"""Mask-guided readout of a lesion embedding from a frozen VLM token map.

The nodule mask (Part 1 output, channel 1 of the crop) is downsampled to the
token grid as an *occupancy* map: the fraction of each token's voxels that
belong to the nodule. Two lesion readouts are produced:

``token_mean``    occupancy-weighted mean of tokens -> (B, C). Lives in the
                  encoder's token space, not the image-text space.
``masked_latent`` occupancy-weighted map passed through the VLM's own image
                  projection -> (B, latent). Lives in the image-text space, so
                  it is directly comparable with concept-text embeddings.

The global (unmasked) latent of the encoded window is also returned; in ``scan`` mode it is
the whole-scan CT-CLIP embedding that arm A (plain CT-CLIP -> risk) needs.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor


def mask_to_token_occupancy(
    mask: Tensor, patch_size: tuple[int, int, int], dilate_tokens: int = 0
) -> Tensor:
    """(B, 1, D, H, W) soft/binary mask -> (B, t, h, w) occupancy in [0, 1].

    ``dilate_tokens`` grows the region by k tokens in each direction to include
    peri-nodular context (the occupancy of added tokens is the max of their
    neighbours, so weights stay in [0, 1]).
    """
    if mask.dim() != 5 or mask.shape[1] != 1:
        raise ValueError(f"mask must be (B, 1, D, H, W), got {tuple(mask.shape)}")
    occ = F.avg_pool3d(mask.float(), kernel_size=patch_size, stride=patch_size)
    if dilate_tokens > 0:
        k = 2 * dilate_tokens + 1
        occ = F.max_pool3d(occ, kernel_size=k, stride=1, padding=dilate_tokens)
    return occ[:, 0]


def masked_token_mean(tokens: Tensor, occ: Tensor, eps: float = 1e-6) -> Tensor:
    """(B, t, h, w, C), (B, t, h, w) -> (B, C)."""
    w = occ.unsqueeze(-1)
    return (tokens * w).sum(dim=(1, 2, 3)) / w.sum(dim=(1, 2, 3)).clamp_min(eps)


def masked_spatial_map(tokens: Tensor, occ: Tensor, eps: float = 1e-6) -> Tensor:
    """Depth-pool tokens with occupancy weights, then gate each (h, w) column.

    Returns (B, h, w, C). Columns with no nodule are zero; the column gate is the
    column's total occupancy normalised to max 1 per sample. With a full mask
    this reduces to CT-CLIP's own depth mean, so the readout is a strict
    generalisation of the global path.
    """
    w = occ.unsqueeze(-1)  # (B, t, h, w, 1)
    col_mass = w.sum(dim=1)  # (B, h, w, 1)
    depth_mean = (tokens * w).sum(dim=1) / col_mass.clamp_min(eps)
    gate = col_mass / col_mass.amax(dim=(1, 2, 3), keepdim=True).clamp_min(eps)
    return depth_mean * gate


@dataclass
class LesionReadout:
    token_mean: Tensor  # (B, C)
    masked_latent: Tensor  # (B, latent), L2-normalised
    global_latent: Tensor  # (B, latent), L2-normalised
    occupancy_sum: Tensor  # (B,) total token occupancy; ~0 means the mask missed the grid


@torch.no_grad()
def readout_from_tokens(vlm, tokens: Tensor, nodule_mask: Tensor, dilate_tokens: int = 0) -> LesionReadout:
    """Readouts for one mask from an already computed token map (lets one encoder pass serve many nodules)."""
    occ = mask_to_token_occupancy(nodule_mask, vlm.patch_size, dilate_tokens)
    if occ.shape[1:] != tokens.shape[1:4]:
        raise RuntimeError(
            f"Token grid {tuple(tokens.shape[1:4])} != mask grid {tuple(occ.shape[1:])}; "
            "check encoder input_shape / patch_size against the window size."
        )
    tokens = tokens.float()
    masked = vlm.project_spatial_map(masked_spatial_map(tokens, occ)).float()
    glob = vlm.project_spatial_map(tokens.mean(dim=1)).float()
    return LesionReadout(
        token_mean=masked_token_mean(tokens, occ),
        masked_latent=F.normalize(masked, dim=-1),
        global_latent=F.normalize(glob, dim=-1),
        occupancy_sum=occ.sum(dim=(1, 2, 3)),
    )


@torch.no_grad()
def lesion_readout(vlm, ct: Tensor, nodule_mask: Tensor, dilate_tokens: int = 0) -> LesionReadout:
    """Encode ``ct`` and read out the lesion in one call."""
    return readout_from_tokens(vlm, vlm.encode_tokens(ct), nodule_mask, dilate_tokens)
