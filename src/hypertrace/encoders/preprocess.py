"""Turn a raw bucket scan + one nodule's instance mask into exactly what CT-CLIP expects.

The bucket images are NOT normalised: int16 Hounsfield units at native, anisotropic
spacing (LIDC slice spacing 0.5 to 3 mm). CT-CLIP was trained on CT-RATE volumes
clipped to [-1000, 1000] HU, scaled by 1000, and resampled to a fixed spacing (z
1.5 mm, y/x 0.75 mm; verify against your checkout). Per scan:

1. ``check_hu_range`` fails if intensities contradict ``hu_already_normalized``.
2. ``normalize_hu`` clips and scales, *before* resampling, so -2048/-3024 out-of-FOV
   padding doesn't bleed into tissue.
3. ``resample_to_spacing`` (trilinear) brings the scan to CT-CLIP spacing.

Per nodule, ``instance == label`` is resampled onto exactly the same grid and kept
soft (it is only used as token-occupancy weights). Then one of two input modes:

``scan``  The whole scan is centre-cropped/padded to the encoder canvas, as CT-RATE
          was. At CT-CLIP spacing the canvas is about 360 mm, close to LIDC's field
          of view, so this is in-distribution. If a nodule falls outside the centred
          window, the window shifts to contain it. One encoder pass is shared by all
          nodules in the same window.
``crop``  A cube of ``crop_size_mm`` centred on the nodule is resized to the encoder
          canvas: a zoomed "grounded lesion crop", as in the overview figure. The
          nodule covers many tokens, but the scale is off-distribution.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor


class IntensityError(ValueError):
    """CT intensities contradict the configured normalisation state."""


def check_hu_range(ct: Tensor | np.ndarray, already_normalized: bool, name: str = "scan") -> None:
    lo, hi = float(ct.min()), float(ct.max())
    if not already_normalized and hi - lo < 50:
        raise IntensityError(f"{name}: CT range [{lo:.3g}, {hi:.3g}] looks already normalised, "
                             "but hu_already_normalized=false.")
    if already_normalized and (abs(lo) > 50 or abs(hi) > 50):
        raise IntensityError(f"{name}: CT range [{lo:.3g}, {hi:.3g}] looks like raw HU, "
                             "but hu_already_normalized=true.")


def normalize_hu(ct: Tensor, hu_min: float = -1000.0, hu_max: float = 1000.0, scale: float = 1000.0,
                 shift: float = 0.0) -> Tensor:
    """(clip(HU, hu_min, hu_max) - shift) / scale. CT-CLIP: [-1000, 1000] / 1000; FMCIB: (HU + 1024) / 3072."""
    out = ct.clamp(hu_min, hu_max)
    return (out - shift) / scale if shift else out / scale


def resampled_shape(shape: Sequence[int], spacing: Sequence[float], target: Sequence[float]) -> tuple[int, ...]:
    if any(s <= 0 for s in spacing):
        raise ValueError(f"Invalid voxel spacing {tuple(spacing)}")
    return tuple(max(1, round(n * s / t)) for n, s, t in zip(shape, spacing, target, strict=True))


def resample_to_spacing(vol: Tensor, spacing: Sequence[float], target: Sequence[float] | None) -> Tensor:
    """(B, C, z, y, x) at ``spacing`` mm -> same physical extent at ``target`` mm (trilinear)."""
    if target is None:
        return vol
    size = resampled_shape(vol.shape[2:], spacing, target)
    if tuple(vol.shape[2:]) == size:
        return vol
    return F.interpolate(vol, size=size, mode="trilinear", align_corners=False)


def prepare_ct(ct: np.ndarray, spacing: Sequence[float], target_spacing: Sequence[float] | None,
               device: torch.device | str = "cpu", hu_already_normalized: bool = False,
               hu_window: tuple[float, float] = (-1000.0, 1000.0), hu_scale: float = 1000.0,
               check_intensity: bool = True, name: str = "scan", hu_shift: float = 0.0) -> Tensor:
    """(z, y, x) raw scan -> (1, 1, Z, Y, X) normalised and resampled."""
    t = torch.from_numpy(np.ascontiguousarray(ct, dtype=np.float32))[None, None].to(device)
    if check_intensity:
        check_hu_range(t, hu_already_normalized, name)
    if not hu_already_normalized:
        t = normalize_hu(t, *hu_window, scale=hu_scale, shift=hu_shift)
    return resample_to_spacing(t, spacing, target_spacing)


def prepare_mask(binary: np.ndarray, out_shape: Sequence[int], device: torch.device | str = "cpu") -> Tensor:
    """(z, y, x) {0,1} -> (1, 1, *out_shape) soft mask on the same grid as ``prepare_ct``."""
    t = torch.from_numpy(np.ascontiguousarray(binary, dtype=np.float32))[None, None].to(device)
    if tuple(t.shape[2:]) != tuple(out_shape):
        t = F.interpolate(t, size=tuple(out_shape), mode="trilinear", align_corners=False)
    return t.clamp_(0, 1)


def mask_bbox(mask: Tensor, thr: float = 0.5) -> tuple[tuple[int, int], ...] | None:
    """[lo, hi) per axis of voxels > thr (falls back to > 0 for tiny masks); None if empty."""
    m = mask[0, 0]
    on = m > thr
    if not on.any():
        on = m > 0
    if not on.any():
        return None
    idx = on.nonzero()
    return tuple((int(idx[:, a].min()), int(idx[:, a].max()) + 1) for a in range(3))


def scan_window(shape: Sequence[int], canvas: Sequence[int], bbox) -> tuple[int, ...]:
    """Start index per axis of a ``canvas`` window: centred (CT-RATE style), shifted to contain ``bbox``."""
    starts = []
    for n, c, (lo, hi) in zip(shape, canvas, bbox, strict=True):
        start = (n - c) // 2  # negative => padding on both sides
        if n > c and not (start <= lo and hi <= start + c):
            start = int(np.clip(round((lo + hi) / 2 - c / 2), 0, n - c))
        starts.append(start)
    return tuple(starts)


def crop_window(vol: Tensor, start: Sequence[int], size: Sequence[int], pad_value: float) -> Tensor:
    """Extract ``size`` voxels from ``start`` (may be negative / overflow; filled with ``pad_value``)."""
    out = torch.full((*vol.shape[:2], *size), pad_value, dtype=vol.dtype, device=vol.device)
    src, dst = [], []
    for n, s, c in zip(vol.shape[2:], start, size, strict=True):
        a, b = max(s, 0), min(s + c, n)
        if a >= b:
            return out
        src.append(slice(a, b))
        dst.append(slice(a - s, b - s))
    out[(..., *dst)] = vol[(..., *src)]
    return out


def center_of_mass(mask: Tensor) -> tuple[float, ...]:
    """Voxel coordinates (z, y, x) of the centre of mass of a (1, 1, z, y, x) soft mask."""
    m = mask[0, 0]
    total = m.sum().clamp_min(1e-12)
    coords = []
    for axis in range(3):
        shape = [1, 1, 1]
        shape[axis] = m.shape[axis]
        idx = torch.arange(m.shape[axis], dtype=m.dtype, device=m.device).reshape(shape)
        coords.append(float((m * idx).sum() / total))
    return tuple(coords)


def centered_crop(vol: Tensor, center: Sequence[float], size: Sequence[int], pad_value: float) -> Tensor:
    """``size`` voxels centred on ``center`` (voxel coordinates), no resizing. Same arithmetic as
    FMCIB's seed-based crop: start = int(centre) - size // 2."""
    start = [int(c) - s // 2 for c, s in zip(center, size, strict=True)]
    return crop_window(vol, start, size, pad_value)


def lesion_crop(vol: Tensor, bbox, size_vox: Sequence[int], out_shape: Sequence[int], pad_value: float) -> Tensor:
    """Cube of ``size_vox`` centred on ``bbox``, resized to ``out_shape``."""
    start = [round((lo + hi) / 2 - c / 2) for (lo, hi), c in zip(bbox, size_vox, strict=True)]
    w = crop_window(vol, start, size_vox, pad_value)
    return F.interpolate(w, size=tuple(out_shape), mode="trilinear", align_corners=False)
