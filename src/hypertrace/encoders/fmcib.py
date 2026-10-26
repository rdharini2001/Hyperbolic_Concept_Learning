"""Frozen FMCIB lesion encoder (Pai et al., "Foundation model for cancer imaging biomarkers",
Nature Machine Intelligence 6:354-367, 2024; code: AIM-Harvard/foundation-cancer-image-biomarker).

A 3D ResNet-50 (MONAI, widen factor 2, 1 input channel) pretrained with self-supervised
contrastive learning on 11,467 lesions from DeepLesion (NIH; no LIDC). It embeds a
50 x 50 x 50 voxel cube at 1 mm isotropic spacing, centred on the lesion, into 4,096
features. DeepLesion pretraining means LIDC is unseen by the encoder (the paper used
LUNA16 only as a downstream task).

Preprocessing, matched to the package's ``fmcib.preprocessing.get_transforms`` (v1.0.0):

* intensity: (HU + 1024) / 3072  (MONAI NormalizeIntensity, subtrahend -1024, divisor 3072).
  We clip to [-1024, 3071] HU first; the original does not clip, which only differs for
  out-of-field-of-view padding (-2048 / -3024), not tissue.
* spacing: 1 mm isotropic, linear interpolation.
* orientation: LPS, then transposed to (z, y, x). Our volumes are RAS-canonical (z, y, x), so
  y and x are flipped (``flip_axes = (1, 2)``).
* crop: 50^3 centred on the seed point; the paper used the centre of mass of the 3D
  contour for LUNA16, which is what extraction uses (the nodule mask's centre of mass).
  Out-of-volume voxels are zero after normalisation (= -1024 HU), as SpatialPad does.

Weights (MIT-licensed code; check the Zenodo record for the weights' terms):
https://zenodo.org/records/10528450/files/model_weights.torch  (scripts/download_weights.sh)
Needs ``monai`` (pip install -e ".[fmcib]").
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch
from torch import Tensor

from hypertrace.encoders.base import LesionEncoder

log = logging.getLogger(__name__)


def build_trunk():
    from monai.networks.nets import resnet50

    return resnet50(pretrained=False, n_input_channels=1, widen_factor=2, conv1_t_stride=2,
                    feed_forward=False, bias_downsample=True)


def _extract_state_dict(ckpt) -> dict[str, Tensor]:
    """The trunk weights inside an FMCIB checkpoint (VISSL 'trunk_state_dict' or Lightning 'state_dict')."""
    if isinstance(ckpt, dict) and "trunk_state_dict" in ckpt:
        sd = ckpt["trunk_state_dict"]
    elif isinstance(ckpt, dict) and "state_dict" in ckpt:
        sd = ckpt["state_dict"]
    else:
        sd = ckpt
    out = {}
    for k, v in sd.items():
        for prefix in ("module.", "model.trunk.", "trunk.", "_feature_blocks."):
            if k.startswith(prefix):
                k = k[len(prefix):]
        if not k.startswith(("model.heads.", "heads.")):
            out[k] = v
    return out


def load_fmcib_state(trunk: torch.nn.Module, checkpoint: str | Path) -> None:
    """Load strictly: unlike the original loader (strict=False + a warning), a missing or
    unexpected key stops extraction instead of silently leaving random weights."""
    path = Path(checkpoint).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"FMCIB weights not found at {path}. Run scripts/download_weights.sh "
                                "(or set encoder.fmcib.checkpoint / HT_FMCIB_CKPT).")
    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
    except Exception:  # older pickles with non-tensor objects; only use the official file
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
    sd = _extract_state_dict(ckpt)
    expected = trunk.state_dict()
    missing = sorted(set(expected) - set(sd))
    unexpected = sorted(set(sd) - set(expected))
    if missing or unexpected:
        raise RuntimeError(f"FMCIB checkpoint {path} does not match the MONAI ResNet-50 trunk: "
                           f"{len(missing)} missing (e.g. {missing[:5]}), {len(unexpected)} unexpected "
                           f"(e.g. {unexpected[:5]}).")
    bad = [k for k in expected if tuple(expected[k].shape) != tuple(sd[k].shape)]
    if bad:
        raise RuntimeError(f"FMCIB checkpoint shape mismatch for {bad[:5]}")
    trunk.load_state_dict(sd, strict=True)
    log.info("Loaded FMCIB weights from %s (%d tensors)", path, len(sd))


class FrozenFMCIB(LesionEncoder):
    def __init__(self, checkpoint: str | None = None, crop_size=(50, 50, 50),
                 target_spacing_mm=(1.0, 1.0, 1.0), flip_axes=(1, 2), feature_dim: int = 4096) -> None:
        super().__init__()
        self.crop_size = tuple(int(v) for v in crop_size)
        self.target_spacing = tuple(float(v) for v in target_spacing_mm)
        self.flip_axes = tuple(int(a) for a in flip_axes)
        self.feature_dim = feature_dim
        self.trunk = build_trunk()
        if not checkpoint:
            raise ValueError("encoder.fmcib.checkpoint is empty; set HT_FMCIB_CKPT (scripts/env.sh).")
        load_fmcib_state(self.trunk, checkpoint)
        self.requires_grad_(False).eval()
        with torch.no_grad():  # self-check: architecture and output size
            out = self.trunk(torch.zeros(1, 1, *self.crop_size))
        if out.shape != (1, feature_dim):
            raise RuntimeError(f"FMCIB trunk returned {tuple(out.shape)}, expected (1, {feature_dim})")

    def train(self, mode: bool = True):  # always eval (BatchNorm statistics frozen)
        return super().train(False)

    def encode(self, crop: Tensor) -> Tensor:
        if tuple(crop.shape[2:]) != self.crop_size:
            raise ValueError(f"crop {tuple(crop.shape[2:])} != {self.crop_size}")
        return self.trunk(crop)
