import torch

from hypertrace.encoders.fake import FakeVLM
from hypertrace.encoders.pooling import (
    lesion_readout,
    mask_to_token_occupancy,
    masked_spatial_map,
    masked_token_mean,
)
from hypertrace.encoders.preprocess import crop_window


def test_occupancy_is_fraction_of_token():
    m = torch.zeros(1, 1, 4, 8, 8)
    m[:, :, :2, :4, :4] = 1  # 2*4*4 = 32 of the token's 4*8*8 = 256 voxels
    occ = mask_to_token_occupancy(m, (4, 8, 8))
    assert occ.shape == (1, 1, 1, 1)
    assert occ.item() == 0.125


def test_masked_mean_only_uses_masked_tokens():
    tokens = torch.zeros(1, 2, 2, 2, 3)
    tokens[0, 0, 0, 0] = torch.tensor([1.0, 2.0, 3.0])
    tokens[0, 1, 1, 1] = 100.0
    occ = torch.zeros(1, 2, 2, 2)
    occ[0, 0, 0, 0] = 1.0
    assert torch.allclose(masked_token_mean(tokens, occ), torch.tensor([[1.0, 2.0, 3.0]]))


def test_full_mask_spatial_map_equals_global_depth_mean():
    tokens = torch.randn(2, 3, 4, 4, 5)
    occ = torch.ones(2, 3, 4, 4)
    assert torch.allclose(masked_spatial_map(tokens, occ), tokens.mean(dim=1), atol=1e-6)


def test_readout_with_fake_vlm():
    vlm = FakeVLM()
    ct = torch.full((1, 1, 20, 40, 40), -0.8)
    mask = torch.zeros_like(ct)
    mask[..., 6:10, 12:20, 12:20] = 1
    ct = ct + 0.85 * mask
    start = (-6, -12, -12)  # centre-pad into the (32, 64, 64) canvas
    canvas = vlm.input_shape
    r = lesion_readout(vlm, crop_window(ct, start, canvas, -1.0), crop_window(mask, start, canvas, 0.0))
    assert r.masked_latent.shape == (1, vlm.latent_dim)
    assert r.token_mean.shape == (1, vlm.token_dim)
    assert r.occupancy_sum.item() > 0
    assert torch.allclose(r.masked_latent.norm(dim=-1), torch.ones(1), atol=1e-5)
