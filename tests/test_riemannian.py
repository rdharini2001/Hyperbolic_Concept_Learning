"""Riemannian optimisation of free concept points on the Poincaré ball."""

import math
from importlib.resources import files

import pytest
import torch
import yaml

from hypertrace.concepts import ConceptHierarchy
from hypertrace.geometry import build_geometry
from hypertrace.geometry import poincare as pm
from hypertrace.models.model import init_concept_tangents
from hypertrace.optim import RiemannianAdamW

torch.manual_seed(0)


def _pts(n=16, d=5, r=0.8):
    v = torch.randn(n, d)
    return v / v.norm(dim=-1, keepdim=True) * torch.rand(n, 1) * r


@pytest.mark.parametrize("c", [0.5, 1.0, 2.0])
def test_expmap_moves_by_riemannian_length(c):
    x = _pts(r=0.8 / math.sqrt(c)).double()
    u = torch.randn_like(x) * 0.1
    y = pm.expmap(x, u, c)
    lam = pm.conformal_factor(x, c).squeeze(-1)
    assert torch.allclose(pm.dist(x, y, c), lam * u.norm(dim=-1), atol=1e-6)
    assert torch.allclose(pm.expmap(x, torch.zeros_like(u), c), x, atol=1e-7)
    assert torch.allclose(pm.expmap(torch.zeros_like(u), u, c), pm.expmap0(u, c), atol=1e-7)


@pytest.mark.parametrize("c", [0.5, 1.0])
def test_parallel_transport_preserves_riemannian_norm(c):
    x, y = _pts(r=0.7 / math.sqrt(c)).double(), _pts(r=0.7 / math.sqrt(c)).double()
    v = torch.randn_like(x)
    pv = pm.transp(x, y, v, c)
    nx = pm.conformal_factor(x, c) * v.norm(dim=-1, keepdim=True)
    ny = pm.conformal_factor(y, c) * pv.norm(dim=-1, keepdim=True)
    assert torch.allclose(nx, ny, atol=1e-8)
    assert torch.allclose(pm.transp(x, x, v, c), v, atol=1e-8)  # transport to itself is the identity


def test_euclidean_manifold_group_is_exactly_adam():
    g = build_geometry("euclidean")
    a = torch.nn.Parameter(torch.randn(6, 4))
    b = torch.nn.Parameter(a.detach().clone())
    target = torch.randn(6, 4)
    o1 = RiemannianAdamW([{"params": [a], "manifold": True}], g, lr=1e-2)
    o2 = torch.optim.Adam([b], lr=1e-2)
    for _ in range(20):
        for p, o in ((a, o1), (b, o2)):
            o.zero_grad()
            ((p - target) ** 3).abs().sum().backward()
            o.step()
    assert torch.allclose(a, b, atol=1e-6)


def test_coordinate_second_moment_steps_like_adam_in_riemannian_length():
    # one step from rest: each coordinate moves by ~lr in Riemannian units, like Adam in the twin
    g = build_geometry("poincare")
    x0 = pm.expmap0(torch.randn(3, 8) * 0.3, 1.0)
    x = torch.nn.Parameter(x0.clone())
    opt = RiemannianAdamW([{"params": [x], "manifold": True}], g, lr=1e-2)
    x.grad = torch.randn_like(x)
    opt.step()
    moved = pm.dist(x0, x.detach(), 1.0)
    assert torch.allclose(moved, torch.full_like(moved, 1e-2 * math.sqrt(8)), rtol=1e-3)


def test_plain_groups_match_torch_adamw():
    g = build_geometry("poincare")
    a = torch.nn.Parameter(torch.randn(5, 3))
    b = torch.nn.Parameter(a.detach().clone())
    o1 = RiemannianAdamW([{"params": [a], "weight_decay": 0.1}], g, lr=5e-3)
    o2 = torch.optim.AdamW([{"params": [b], "weight_decay": 0.1}], lr=5e-3)
    for _ in range(20):
        for p, o in ((a, o1), (b, o2)):
            o.zero_grad()
            (p.sin() ** 2).sum().backward()
            o.step()
    assert torch.allclose(a, b, atol=1e-6)


def test_riemannian_adam_converges_inside_ball():
    g = build_geometry("poincare")
    target = pm.expmap0(torch.tensor([[1.5, -1.0, 0.5]]), 1.0)
    x = torch.nn.Parameter(pm.expmap0(torch.tensor([[-1.0, 1.0, 0.0]]), 1.0))
    opt = RiemannianAdamW([{"params": [x], "manifold": True}], g, lr=0.05)
    for _ in range(400):
        opt.zero_grad()
        pm.pairwise_dist(x, target, 1.0).sum().backward()
        opt.step()
        assert x.norm() < 1
    assert pm.dist(x, target, 1.0).item() < 0.05


@pytest.mark.parametrize("geometry", ["poincare", "euclidean"])
def test_matches_geoopt_riemannian_adam(geometry):
    """Independent implementation check against geoopt (Kochurov et al., 2020)."""
    geoopt = pytest.importorskip("geoopt")

    class ExactBall(geoopt.PoincareBall):  # geoopt retracts with x + u; give it the exact exp map
        def retr(self, x, u, *, dim=-1):
            return self.expmap(x, u, dim=dim)

    manifold = ExactBall(c=1.0) if geometry == "poincare" else geoopt.Euclidean(ndim=1)
    g = build_geometry(geometry)
    target = pm.expmap0(torch.randn(4, 3), 1.0).double()
    start = pm.expmap0(torch.randn(4, 3) * 0.5, 1.0).double()
    ours = torch.nn.Parameter(start.clone())
    ref = geoopt.ManifoldParameter(start.clone(), manifold=manifold)
    # geoopt: per-point second moment on the ball, per-coordinate for its Euclidean manifold
    o1 = RiemannianAdamW([{"params": [ours], "manifold": True}], g, lr=1e-2,
                         second_moment="point" if geometry == "poincare" else "coordinate")
    o2 = geoopt.optim.RiemannianAdam([ref], lr=1e-2)
    for _ in range(50):
        for p, o in ((ours, o1), (ref, o2)):
            o.zero_grad()
            pm.dist(p, target, 1.0).pow(2).sum().backward()
            o.step()
    assert torch.allclose(ours, ref.detach(), atol=1e-6)


def test_tree_init_puts_general_concepts_near_origin():
    h = ConceptHierarchy.from_config(yaml.safe_load((files("hypertrace") / "configs/concepts/lidc.yaml").read_text()))
    v = init_concept_tangents(h, 16, "tree")
    norms = v.norm(dim=-1)
    for n in h.nodes:
        if n.parent:
            assert norms[h.index(n.name)] > norms[h.index(n.parent)]
    with pytest.raises(ValueError, match="prompt embeddings"):
        init_concept_tangents(h, 16, "text", None)
    t = init_concept_tangents(h, 16, "text", torch.randn(len(h), 32))
    assert torch.allclose(t.norm(dim=-1), norms, atol=1e-5)  # same radii, text-derived directions
