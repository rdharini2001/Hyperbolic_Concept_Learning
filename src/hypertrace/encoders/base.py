"""Encoder interfaces.

Two kinds of frozen encoder are supported:

* ``CTVisionLanguageModel`` (CT-CLIP): a whole-scan or zoomed-crop token map, read out
  at the nodule with the mask, plus a text encoder for the concept prompts.
* ``LesionEncoder`` (FMCIB): one embedding per lesion-centred cube at the encoder's own
  scale; no text encoder.

Part 2 needs two things from a 3D CT VLM:

1. a *token map* for a CT volume, shape (B, t, h, w, C), so the nodule mask can
   decide which tokens describe the lesion; and
2. the VLM's own projection into its image-text latent space, plus a text
   encoder into the same space.

Hiding CT-CLIP behind this interface means the rest of the code (and the unit
tests, via ``FakeVLM``) never imports CT-CLIP, and swapping in another 3D VLM
later (e.g. a newer CT foundation model) is a new subclass, not a refactor.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from torch import Tensor, nn


class CTVisionLanguageModel(nn.Module, ABC):
    kind = "tokens"
    #: expected (D, H, W) of the CT fed to the image encoder
    input_shape: tuple[int, int, int]
    #: voxels per token along (D, H, W)
    patch_size: tuple[int, int, int]
    #: channel dim of encoder tokens
    token_dim: int
    #: dim of the shared image-text latent space
    latent_dim: int

    @property
    def token_grid(self) -> tuple[int, int, int]:
        return tuple(s // p for s, p in zip(self.input_shape, self.patch_size, strict=True))

    @abstractmethod
    def encode_tokens(self, ct: Tensor) -> Tensor:
        """(B, 1, D, H, W) normalised CT -> (B, t, h, w, C) token map."""

    @abstractmethod
    def project_spatial_map(self, spatial_map: Tensor) -> Tensor:
        """(B, h, w, C) depth-pooled token map -> (B, latent_dim) image latent (un-normalised).

        This mirrors how the VLM turns its tokens into its global image latent,
        so a *masked* spatial map lands in the same image-text space.
        """

    @abstractmethod
    def encode_text(self, prompts: list[str]) -> Tensor:
        """List of N prompts -> (N, latent_dim) text latents (un-normalised)."""


class LesionEncoder(nn.Module, ABC):
    """A frozen encoder that embeds a fixed-size cube around ONE lesion (no token map, no text).

    Extraction resamples the scan to ``target_spacing`` (z, y, x mm), normalises intensities with
    the encoder's own HU settings, cuts ``crop_size`` voxels centred on the nodule mask's centre
    of mass, flips ``flip_axes`` (spatial axes of the (z, y, x) crop) to match the encoder's
    training orientation, and calls ``encode``.
    """

    kind = "lesion"
    #: crop size in voxels (z, y, x)
    crop_size: tuple[int, int, int]
    #: voxel spacing the encoder was trained at (z, y, x) mm
    target_spacing: tuple[float, float, float]
    #: spatial axes (0 = z, 1 = y, 2 = x) to flip after cropping our RAS-canonical (z, y, x) arrays
    flip_axes: tuple[int, ...] = ()
    #: dimension of the output embedding
    feature_dim: int

    @abstractmethod
    def encode(self, crop: Tensor) -> Tensor:
        """(B, 1, *crop_size) normalised crop -> (B, feature_dim) embedding."""
