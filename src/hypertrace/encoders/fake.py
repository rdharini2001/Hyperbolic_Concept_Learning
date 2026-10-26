"""Deterministic stand-in for CT-CLIP.

Same interface and token layout, tiny and random. Used by the tests and for a
dry run of the full pipeline on a login node before the real weights are set up.
Results with it are meaningless by design.
"""

from __future__ import annotations

import hashlib

import torch
from torch import Tensor, nn

from hypertrace.encoders.base import CTVisionLanguageModel, LesionEncoder


class FakeVLM(CTVisionLanguageModel):
    def __init__(
        self,
        input_shape: tuple[int, int, int] = (32, 64, 64),
        patch_size: tuple[int, int, int] = (4, 8, 8),
        token_dim: int = 32,
        latent_dim: int = 64,
        seed: int = 0,
    ) -> None:
        super().__init__()
        self.input_shape = tuple(input_shape)
        self.patch_size = tuple(patch_size)
        self.token_dim = token_dim
        self.latent_dim = latent_dim
        g = torch.Generator().manual_seed(seed)
        self.patchify = nn.Conv3d(1, token_dim, kernel_size=patch_size, stride=patch_size)
        h, w = self.token_grid[1:]
        self.to_visual_latent = nn.Linear(h * w * token_dim, latent_dim)
        for p in self.parameters():
            with torch.no_grad():
                p.copy_(torch.randn(p.shape, generator=g) * 0.05)
        self.requires_grad_(False).eval()

    def encode_tokens(self, ct: Tensor) -> Tensor:
        return self.patchify(ct).permute(0, 2, 3, 4, 1)  # (B, t, h, w, C)

    def project_spatial_map(self, spatial_map: Tensor) -> Tensor:
        return self.to_visual_latent(spatial_map.reshape(spatial_map.shape[0], -1))

    def encode_text(self, prompts: list[str]) -> Tensor:
        out = []
        for p in prompts:
            s = int(hashlib.sha256(p.encode()).hexdigest()[:8], 16)
            out.append(torch.randn(self.latent_dim, generator=torch.Generator().manual_seed(s)))
        return torch.stack(out)


class FakeLesionEncoder(LesionEncoder):
    """Tiny random stand-in for FMCIB (tests / dry runs): conv + global average pool."""

    def __init__(self, crop_size=(16, 16, 16), target_spacing_mm=(2.0, 2.0, 2.0), feature_dim: int = 48,
                 flip_axes=(1, 2), seed: int = 0) -> None:
        super().__init__()
        self.crop_size = tuple(int(v) for v in crop_size)
        self.target_spacing = tuple(float(v) for v in target_spacing_mm)
        self.flip_axes = tuple(int(a) for a in flip_axes)
        self.feature_dim = feature_dim
        g = torch.Generator().manual_seed(seed)
        self.conv = nn.Conv3d(1, feature_dim, kernel_size=3, padding=1)
        for p in self.parameters():
            with torch.no_grad():
                p.copy_(torch.randn(p.shape, generator=g) * 0.2)
        self.requires_grad_(False).eval()

    def encode(self, crop: Tensor) -> Tensor:
        if tuple(crop.shape[2:]) != self.crop_size:
            raise ValueError(f"crop {tuple(crop.shape[2:])} != {self.crop_size}")
        return torch.relu(self.conv(crop)).mean(dim=(2, 3, 4))
