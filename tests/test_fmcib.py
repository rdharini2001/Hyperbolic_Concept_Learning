"""FMCIB wrapper: architecture, strict checkpoint loading, preprocessing conventions."""

import numpy as np
import pytest
import torch

monai = pytest.importorskip("monai")

from hypertrace.encoders.fmcib import FrozenFMCIB, build_trunk  # noqa: E402
from hypertrace.encoders.preprocess import center_of_mass, centered_crop, normalize_hu  # noqa: E402


@pytest.fixture(scope="module")
def trunk_sd():
    torch.manual_seed(0)
    return build_trunk().state_dict()


@pytest.mark.parametrize("layout", ["vissl", "lightning"])
def test_loads_both_checkpoint_layouts(tmp_path, trunk_sd, layout):
    ckpt = ({"trunk_state_dict": trunk_sd} if layout == "vissl"
            else {"state_dict": {f"model.trunk.{k}": v for k, v in trunk_sd.items()}})
    torch.save(ckpt, tmp_path / "w.torch")
    enc = FrozenFMCIB(checkpoint=str(tmp_path / "w.torch"))
    out = enc.encode(torch.zeros(2, 1, 50, 50, 50))
    assert out.shape == (2, 4096)
    assert torch.equal(enc.trunk.state_dict()["conv1.weight"], trunk_sd["conv1.weight"])
    assert not enc.training and not any(p.requires_grad for p in enc.parameters())


def test_rejects_wrong_checkpoint(tmp_path, trunk_sd):
    bad = {k: v for k, v in trunk_sd.items() if not k.startswith("layer4")}
    torch.save({"trunk_state_dict": bad}, tmp_path / "bad.torch")
    with pytest.raises(RuntimeError, match="missing"):
        FrozenFMCIB(checkpoint=str(tmp_path / "bad.torch"))


def test_fmcib_intensity_and_crop_conventions():
    hu = torch.tensor([-3024.0, -1024.0, 0.0, 2048.0])
    x = normalize_hu(hu, -1024.0, 3071.0, scale=3072.0, shift=-1024.0)
    assert torch.allclose(x, torch.tensor([0.0, 0.0, 1024 / 3072, 1.0]))  # (HU + 1024) / 3072, padding clipped
    vol = torch.zeros(1, 1, 60, 60, 60)
    vol[0, 0, 20:25, 30:33, 40:41] = 1
    com = center_of_mass(vol)
    assert np.allclose(com, (22.0, 31.0, 40.0))
    crop = centered_crop(vol, com, (50, 50, 50), 0.0)
    assert crop.shape[2:] == (50, 50, 50) and crop.sum() == vol.sum()  # nodule fully inside, centred
    assert crop[0, 0, 25 - 2 : 25 + 3, 25 - 1 : 25 + 2, 25].sum() == vol.sum()
