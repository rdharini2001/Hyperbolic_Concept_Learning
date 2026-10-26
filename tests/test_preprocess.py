import numpy as np
import pytest
import torch

from hypertrace.data.volumes import check_spacing, load_volume
from hypertrace.encoders.preprocess import (
    IntensityError,
    check_hu_range,
    crop_window,
    lesion_crop,
    mask_bbox,
    prepare_ct,
    prepare_mask,
    resample_to_spacing,
    scan_window,
)


def _sphere(shape=(10, 20, 20), spacing=(2.5, 0.7, 0.7), radius_mm=5.0, center=None):
    center = center or [n / 2 for n in shape]
    zz, yy, xx = np.meshgrid(*[np.arange(n) for n in shape], indexing="ij")
    d = np.sqrt(sum(((g - c) * s) ** 2 for g, c, s in zip((zz, yy, xx), center, spacing, strict=True)))
    return d <= radius_mm


def test_resample_preserves_physical_extent():
    out = resample_to_spacing(torch.zeros(1, 1, 10, 20, 20), (2.5, 0.7, 0.7), (1.5, 0.75, 0.75))
    assert out.shape[2:] == (17, 19, 19)  # round(10*2.5/1.5), round(20*0.7/0.75)


def test_mask_volume_preserved_and_aligned_with_ct():
    sp, target = (2.5, 0.7, 0.7), (1.0, 0.5, 0.5)
    nod = _sphere(spacing=sp)
    ct = prepare_ct(np.where(nod, 40, -800).astype(np.int16), sp, target)
    m = prepare_mask(nod, ct.shape[2:])
    assert m.shape == ct.shape  # same grid by construction
    assert m.sum().item() * np.prod(target) == pytest.approx(nod.sum() * np.prod(sp), rel=0.1)
    assert ct[m > 0.9].mean() > ct[m < 0.1].mean()  # mask sits on the bright nodule


def test_hu_clipped_before_resampling():
    raw = np.full((10, 20, 20), -800, np.int16)
    raw[:, :, :2] = -2048
    ct = prepare_ct(raw, (2.5, 0.7, 0.7), (1.5, 0.75, 0.75))
    assert ct.min() >= -1.0 and ct.max() <= 1.0


def test_intensity_guard_catches_both_mistakes():
    raw = torch.tensor([-1000.0, 40.0, 300.0])
    check_hu_range(raw, already_normalized=False)
    with pytest.raises(IntensityError, match="raw HU"):
        check_hu_range(raw, already_normalized=True)
    with pytest.raises(IntensityError, match="already normalised"):
        check_hu_range(raw / 1000, already_normalized=False)


def test_scan_window_centres_then_shifts_to_contain_nodule():
    assert scan_window((100, 50, 50), (40, 64, 64), ((45, 55), (10, 20), (10, 20))) == (30, -7, -7)
    start = scan_window((100, 50, 50), (40, 64, 64), ((2, 6), (10, 20), (10, 20)))  # near the top
    assert start[0] == 0
    start = scan_window((100, 50, 50), (40, 64, 64), ((90, 98), (10, 20), (10, 20)))  # near the bottom
    assert start[0] == 60 and 90 >= start[0] and 98 <= start[0] + 40


def test_crop_window_pads_and_preserves_content():
    v = torch.arange(4 * 5 * 6, dtype=torch.float32).reshape(1, 1, 4, 5, 6)
    w = crop_window(v, (-1, 1, 2), (6, 3, 3), pad_value=-9)
    assert w.shape == (1, 1, 6, 3, 3)
    assert (w[0, 0, 0] == -9).all() and (w[0, 0, 5] == -9).all()
    assert torch.equal(w[0, 0, 1:5], v[0, 0, :, 1:4, 2:5])


def test_lesion_crop_centres_nodule():
    nod = torch.from_numpy(_sphere((30, 40, 40), (1, 1, 1), 3, center=[8, 30, 12]).astype(np.float32))[None, None]
    out = lesion_crop(nod, mask_bbox(nod), (12, 12, 12), (24, 24, 24), 0.0)
    (z0, z1), (y0, y1), (x0, x1) = mask_bbox(out)
    assert abs((z0 + z1) / 2 - 12) <= 1.5 and abs((y0 + y1) / 2 - 12) <= 1.5 and abs((x0 + x1) / 2 - 12) <= 1.5


@pytest.mark.parametrize("flip_xy", [False, True])
def test_load_volume_canonical_zyx(tmp_path, flip_xy):
    nib = pytest.importorskip("nibabel")
    zyx = np.zeros((4, 5, 6), np.int16)
    zyx[1, 2, 3] = 7
    sx, sy, sz = 0.7, 0.8, 2.5
    aff = np.diag([-sx if flip_xy else sx, -sy if flip_xy else sy, sz, 1.0])  # LPS-like vs RAS
    nib.save(nib.Nifti1Image(zyx.transpose(2, 1, 0), aff), tmp_path / "v.nii.gz")
    v = load_volume(tmp_path / "v.nii.gz")
    assert v.spacing == pytest.approx((sz, sy, sx)) and v.array.shape == (4, 5, 6)
    assert tuple(np.argwhere(v.array == 7)[0]) == ((1, 2, 2) if flip_xy else (1, 2, 3))
    check_spacing(v, (sz, sy, sx), "v")
    with pytest.raises(ValueError, match="spacing"):
        check_spacing(v, (1.0, sy, sx), "v")
