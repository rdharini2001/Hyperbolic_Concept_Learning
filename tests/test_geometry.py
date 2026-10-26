import pytest
import torch

from hypertrace.geometry import EuclideanGeometry, PoincareGeometry
from hypertrace.geometry import poincare as pm

torch.manual_seed(0)


@pytest.mark.parametrize("c", [0.5, 1.0, 2.0])
def test_exp_log_inverse(c):
    v = torch.randn(32, 8) * 0.5
    assert torch.allclose(pm.logmap0(pm.expmap0(v, c), c), v, atol=1e-4)


@pytest.mark.parametrize("c", [0.5, 1.0, 2.0])
def test_points_stay_inside_ball(c):
    x = pm.expmap0(torch.randn(64, 8) * 50, c)
    assert (x.norm(dim=-1) < 1 / c**0.5).all()


@pytest.mark.parametrize("c", [0.5, 1.0, 2.0])
def test_pairwise_matches_mobius_distance(c):
    # Deterministic inputs, kept within the region the model can reach (tangent norm <= clip_r = 2):
    # very close to the boundary the Mobius form saturates at its artanh clamp, the arcosh form doesn't.
    g = torch.Generator().manual_seed(0)
    def pts(n):
        v = torch.randn(n, 4, generator=g)
        return pm.expmap0(v / v.norm(dim=-1, keepdim=True) * torch.rand(n, 1, generator=g) * 2.0, c)
    x, y = pts(5), pts(7)
    a = pm.pairwise_dist(x, y, c)
    b = pm.dist(x.unsqueeze(1), y.unsqueeze(0), c)
    assert torch.allclose(a, b, atol=1e-3)


def test_distance_properties():
    x = pm.expmap0(torch.randn(10, 4), 1.0)
    d = pm.pairwise_dist(x, x, 1.0)
    assert torch.allclose(d, d.T, atol=1e-4)
    assert d.diagonal().abs().max() < 1e-2
    assert (d >= 0).all()


def test_dist0_matches_origin_distance():
    x = pm.expmap0(torch.randn(6, 4) * 0.5, 1.0)
    assert torch.allclose(pm.dist0(x, 1.0), pm.dist(torch.zeros_like(x), x, 1.0), atol=1e-4)


@pytest.mark.parametrize("geom", [PoincareGeometry(), EuclideanGeometry()])
def test_entailment_cone_contains_radial_child(geom):
    # A child further out along the parent's own ray is inside the parent's cone.
    parent = geom.to_space(torch.tensor([[0.6, 0.0, 0.0]]))
    child = geom.to_space(torch.tensor([[1.2, 0.0, 0.0]]))
    assert geom.entailment_violation(parent, child).item() < 1e-4
    # A point in the opposite direction is outside it.
    other = geom.to_space(torch.tensor([[-1.2, 0.0, 0.0]]))
    assert geom.entailment_violation(parent, other).item() > 0.1


def test_gradients_finite_near_boundary():
    v = (torch.randn(16, 8) * 20).requires_grad_()
    g = PoincareGeometry()
    x = g.to_space(v)
    loss = g.pairwise_dist(x, x.detach().flip(0)).sum() + g.entailment_violation(x[:8], x[8:]).sum()
    loss.backward()
    assert torch.isfinite(v.grad).all()


def test_learned_curvature_is_bounded():
    g = PoincareGeometry(curvature=1.0, learn_curvature=True, curvature_bounds=(0.5, 2.0))
    with torch.no_grad():
        g.log_c.fill_(10.0)
    assert g.c.item() == pytest.approx(2.0)
