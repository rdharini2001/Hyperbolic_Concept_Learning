"""Frozen CT-CLIP wrapper.

Requires the CT-CLIP repository (https://github.com/ibrahimethemhamamci/CT-CLIP)
installed into your environment (no root needed):

    pip install -e /path/to/CT-CLIP/transformer_maskgit
    pip install -e /path/to/CT-CLIP/CT_CLIP

and the pretrained checkpoint plus the
``microsoft/BiomedVLP-CXR-BERT-specialized`` tokenizer/model downloaded on a
login node (compute nodes are often offline; see README).

VERIFY AGAINST YOUR CHECKOUT. The constructor arguments and the forward path
below follow CT-CLIP's public training script as I remember it, but I could
not open the repository from the environment where this was written:

* CTViT(dim=512, codebook_size=8192, image_size=480, patch_size=20,
        temporal_patch_size=10, spatial_depth=4, temporal_depth=4,
        dim_head=32, heads=8)
* CTCLIP(dim_image=294912, dim_text=768, dim_latent=512, ...)
  where 294912 = 24 * 24 * 512, i.e. the image latent is a linear map of the
  depth-averaged (h, w, C) token map, flattened.
* image path: tokens = visual_transformer(x, return_encoded_tokens=True)
  -> (B, t, h, w, C); mean over t; flatten; to_visual_latent.
* text path: CLS token of the BERT output -> to_text_latent.

``_self_check`` runs once at load time and fails loudly if any of these
assumptions are wrong, instead of producing silently wrong embeddings.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from hypertrace.encoders.base import CTVisionLanguageModel

log = logging.getLogger(__name__)

DEFAULT_CTVIT_KWARGS: dict[str, Any] = dict(
    dim=512,
    codebook_size=8192,
    image_size=480,
    patch_size=20,
    temporal_patch_size=10,
    spatial_depth=4,
    temporal_depth=4,
    dim_head=32,
    heads=8,
)
DEFAULT_CTCLIP_KWARGS: dict[str, Any] = dict(
    dim_image=294912,
    dim_text=768,
    dim_latent=512,
    extra_latent_projection=False,
    use_mlm=False,
    downsample_image_embeds=False,
    use_all_token_embeds=False,
)


# Buffers that newer `transformers` no longer register (e.g. BERT position_ids since ~4.31).
# A checkpoint saved with an older version still contains them; they are recomputed, not learned.
_IGNORABLE_KEY_SUFFIXES = ("position_ids",)


def load_ctclip_state(model: torch.nn.Module, path: Path) -> None:
    """Load the official CT-CLIP checkpoint robustly across library versions.

    * torch >= 2.6 defaults to ``weights_only=True``; the official file may contain
      non-tensor objects, so it is loaded with ``weights_only=False``. Only load
      checkpoints from a source you trust (here: the gated CT-RATE repository).
    * Accepts either a bare state dict or one wrapped as ``{"model"|"state_dict": ...}``.
    * Ignores only known-harmless buffer mismatches; any other missing or unexpected
      key is an error, so a wrong checkpoint can't load silently with random weights.
    """
    sd = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(sd, dict):
        for key in ("model", "state_dict"):
            if key in sd and isinstance(sd[key], dict):
                sd = sd[key]
                break
    sd = {k.removeprefix("module."): v for k, v in sd.items()}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    bad_missing = [k for k in missing if not k.endswith(_IGNORABLE_KEY_SUFFIXES)]
    bad_unexpected = [k for k in unexpected if not k.endswith(_IGNORABLE_KEY_SUFFIXES)]
    if bad_missing or bad_unexpected:
        raise RuntimeError(
            f"CT-CLIP checkpoint {path} does not match the model.\n"
            f"  missing ({len(bad_missing)}): {bad_missing[:8]}\n"
            f"  unexpected ({len(bad_unexpected)}): {bad_unexpected[:8]}\n"
            "Check the checkpoint file (CT-CLIP_v2.pt), the CTViT/CTCLIP constructor arguments, and that "
            "vector-quantize-pytorch==1.1.2 is installed (its parameter names are part of the state dict)."
        )
    log.info("Loaded CT-CLIP weights from %s (%d tensors; ignored %d version-specific buffers)",
             path, len(sd), len(missing) + len(unexpected))


class FrozenCTCLIP(CTVisionLanguageModel):
    def __init__(
        self,
        checkpoint: str,
        text_model: str = "microsoft/BiomedVLP-CXR-BERT-specialized",
        input_shape: tuple[int, int, int] = (240, 480, 480),
        patch_size: tuple[int, int, int] = (10, 20, 20),
        token_dim: int = 512,
        latent_dim: int = 512,
        max_text_len: int = 512,
        ctvit_kwargs: dict[str, Any] | None = None,
        ctclip_kwargs: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        # Imported lazily so the rest of the package (and the tests) never need CT-CLIP.
        try:
            from ct_clip import CTCLIP
            from transformer_maskgit import CTViT
        except ImportError as e:  # pragma: no cover - depends on user env
            raise ImportError(
                "CT-CLIP is not installed. `pip install -e CT-CLIP/transformer_maskgit` and "
                "`pip install -e CT-CLIP/CT_CLIP` inside your venv (see README)."
            ) from e
        from transformers import BertModel, BertTokenizer

        self.input_shape = tuple(input_shape)
        self.patch_size = tuple(patch_size)
        self.token_dim = token_dim
        self.latent_dim = latent_dim
        self.max_text_len = max_text_len

        self.tokenizer = BertTokenizer.from_pretrained(text_model, do_lower_case=True)
        text_encoder = BertModel.from_pretrained(text_model)
        image_encoder = CTViT(**{**DEFAULT_CTVIT_KWARGS, **(ctvit_kwargs or {})})
        self.clip = CTCLIP(
            image_encoder=image_encoder,
            text_encoder=text_encoder,
            **{**DEFAULT_CTCLIP_KWARGS, **(ctclip_kwargs or {})},
        )
        ckpt = Path(checkpoint).expanduser()
        if not ckpt.is_file():
            raise FileNotFoundError(f"CT-CLIP checkpoint not found: {ckpt}")
        load_ctclip_state(self.clip, ckpt)
        self.clip.eval().requires_grad_(False)
        self._self_check()

    # ------------------------------------------------------------------ API
    def encode_tokens(self, ct: Tensor) -> Tensor:
        # CTViT treats depth as the "temporal" axis: input (B, C, T=D, H, W).
        return self.clip.visual_transformer(ct, return_encoded_tokens=True)

    def project_spatial_map(self, spatial_map: Tensor) -> Tensor:
        flat = spatial_map.reshape(spatial_map.shape[0], -1)
        return self.clip.to_visual_latent(flat.to(self.clip.to_visual_latent.weight.dtype))

    def encode_text(self, prompts: list[str]) -> Tensor:
        dev = next(self.clip.parameters()).device
        tok = self.tokenizer(
            prompts, return_tensors="pt", padding="max_length", truncation=True, max_length=self.max_text_len
        ).to(dev)
        out = self.clip.text_transformer(tok.input_ids, attention_mask=tok.attention_mask)
        cls = out[0][:, 0, :]
        return self.clip.to_text_latent(cls)

    # ------------------------------------------------------------ safeguard
    @torch.no_grad()
    def _self_check(self) -> None:
        expected_in = self.clip.to_visual_latent.weight.shape[1]
        grid = self.token_grid
        if expected_in != grid[1] * grid[2] * self.token_dim:
            raise RuntimeError(
                f"to_visual_latent expects {expected_in} inputs but token grid {grid} x {self.token_dim} "
                f"gives {grid[1] * grid[2] * self.token_dim}. The CT-CLIP forward path differs from what "
                "this wrapper assumes; update project_spatial_map / patch_size."
            )
        log.info("CT-CLIP loaded: token grid %s, token dim %d, latent %d", grid, self.token_dim, self.latent_dim)
