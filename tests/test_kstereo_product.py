"""κ-stereographic geometry (sphere, learned curvature) and the per-group product model."""

import math

import pytest
import torch

from hypertrace.concepts import ConceptHierarchy
from hypertrace.concepts.ontology import load_concept_tree
from hypertrace.config import default_config, load_config
from hypertrace.geometry import build_geometry, curvature_of
from hypertrace.geometry import kstereo as ks
from hypertrace.geometry import poincare as pm
from hypertrace.models.model import build_model
from hypertrace.models.product import split_dims
from hypertrace.optim import RiemannianAdamW

torch.manual_seed(0)
X, Y, V = torch.randn(6, 4) * 0.25, torch.randn(6, 4) * 0.25, torch.randn(6, 4) * 0.3


@pytest.mark.parametrize("c", [1.0, 0.5, 2.0])
def test_negative_kappa_is_the_poincare_ball(c):
    k = -c
    X, Y = globals()["X"] * 0.5 / math.sqrt(c), globals()["Y"] * 0.5 / math.sqrt(c)  # well inside the ball
    assert torch.allclose(ks.dist(X, Y, k), pm.dist(X, Y, c), atol=1e-6)
    assert torch.allclose(ks.expmap0(V, k), pm.expmap0(V, c), atol=1e-6)
    assert torch.allclose(ks.logmap0(X, k), pm.logmap0(X, c), atol=1e-6)
    assert torch.allclose(ks.expmap(X, V, k), pm.expmap(X, V, c), atol=1e-6)
    assert torch.allclose(ks.transp(X, Y, V, k), pm.transp(X, Y, V, c), atol=1e-6)
    assert torch.allclose(ks.oxy_angle(X, Y, k), pm.oxy_angle(X, Y, c), atol=1e-5)
    if c == 1.0:  # cone width: absolute units (continuous through κ = 0); equals Ganea's at κ = -1
        assert torch.allclose(ks.half_aperture(X, k), pm.half_aperture(X, c), atol=1e-6)


def test_zero_kappa_is_flat_and_continuous():
    assert torch.allclose(ks.dist(X, Y, 0.0), 2 * (X - Y).norm(dim=-1), atol=1e-6)  # stereographic scale 2
    assert torch.allclose(ks.expmap0(V, 0.0), V)
    for small in (1e-4, -1e-4, 9e-4, 1.1e-3, -1.1e-3):  # Taylor branch and exact branch agree at the seam
        assert torch.allclose(ks.dist(X, Y, small), ks.dist(X, Y, 0.0), atol=5e-4)


def test_positive_kappa_is_the_sphere():
    def inv_stereo(x, k):  # point of the sphere of radius 1/sqrt(k) in R^(d+1)
        r = 1 / math.sqrt(k)
        n2 = (x * x).sum(-1, keepdim=True) / r**2
        return r * torch.cat([2 * x / r / (1 + n2), (1 - n2) / (1 + n2)], -1)

    for k in (1.0, 0.25, 4.0):
        p, q = inv_stereo(X, k), inv_stereo(Y, k)
        r = 1 / math.sqrt(k)
        great_circle = r * torch.acos(((p * q).sum(-1) / r**2).clamp(-1, 1))
        assert torch.allclose(ks.dist(X, Y, k), great_circle, atol=1e-5)
    # exp_0 never reaches the antipode (at infinity), however long the tangent vector
    far = ks.expmap0(torch.tensor([[50.0, 0.0]]), 1.0)
    assert torch.isfinite(far).all() and ks.dist0(far, 1.0).item() < math.pi


@pytest.mark.parametrize("k", [1.0, -0.7, 0.3])
def test_matches_geoopt_stereographic(k):
    geoopt = pytest.importorskip("geoopt")
    M = geoopt.Stereographic(k=k)
    assert torch.allclose(M.dist(X, Y), ks.dist(X, Y, k), atol=1e-6)
    assert torch.allclose(M.expmap(X, V), ks.expmap(X, V, k), atol=1e-6)
    assert torch.allclose(M.transp(X, Y, V), ks.transp(X, Y, V, k), atol=1e-6)
    assert torch.allclose(M.egrad2rgrad(X, V), ks.egrad2rgrad(X, V, k), atol=1e-6)


def test_learned_curvature_has_gradients_through_zero():
    g = build_geometry("kstereo", curvature=0.0, learn_curvature=True)
    for start in (0.0, 5e-4, -0.2, 0.2):
        with torch.no_grad():
            g.kappa_raw.fill_(start)
        g.kappa_raw.grad = None
        g.pairwise_dist(X, Y).sum().backward()
        assert torch.isfinite(g.kappa_raw.grad) and g.kappa_raw.grad != 0
    assert curvature_of(build_geometry("poincare", curvature=0.5)) == -0.5
    assert curvature_of(build_geometry("sphere", curvature=2.0)) == 2.0
    with pytest.raises(ValueError, match="curvature > 0"):
        build_geometry("sphere", curvature=-1.0)


@pytest.mark.parametrize("k", [1.0, -0.5])
def test_riemannian_adam_matches_geoopt_on_stereographic(k):
    geoopt = pytest.importorskip("geoopt")

    class Exact(geoopt.Stereographic):
        def retr(self, x, u, *, dim=-1):
            return self.expmap(x, u, dim=dim)

    g = build_geometry("kstereo", curvature=k)
    target = ks.expmap0(torch.randn(4, 3) * 0.5, k).double()
    start = ks.expmap0(torch.randn(4, 3) * 0.5, k).double()
    ours = torch.nn.Parameter(start.clone())
    ref = geoopt.ManifoldParameter(start.clone(), manifold=Exact(k=k))
    o1 = RiemannianAdamW([{"params": [ours], "manifold": True}], g, lr=1e-2, second_moment="point")
    o2 = geoopt.optim.RiemannianAdam([ref], lr=1e-2)
    for _ in range(50):
        for p, o in ((ours, o1), (ref, o2)):
            o.zero_grad()
            ks.dist(p, target, k).pow(2).sum().backward()
            o.step()
    assert torch.allclose(ours, ref.detach(), atol=1e-6)


# ------------------------------------------------------------------ product model
def _product(assignment="matched", dim=20, **sets):
    sets = [f"{k}={v}" for k, v in sets.items()]
    cfg = load_config(default_config("train"), ["geometry.name=product", f"geometry.product.assignment={assignment}",
                                                 f"model.projector.embed_dim={dim}", *sets])
    h = ConceptHierarchy.from_config(load_concept_tree(cfg["concepts_file"]))

    class Data:
        image_dim, text_latents = 32, None

    return build_model(cfg, h, Data()), h


def test_product_factors_follow_the_assignment():
    m, h = _product("matched", 20)
    info = {f["group"]: f for f in m.factor_info()}
    assert list(info) == ["size", "nodule_type", "shape", "margin", "calcification"]  # unscored groups skipped
    assert [info[g]["geometry"] for g in info] == ["euclidean", "poincare", "sphere", "euclidean", "poincare"]
    assert sum(f["dim"] for f in info.values()) == 20 and not info["shape"]["cones"] and info["nodule_type"]["cones"]
    swapped, _ = _product("swapped", 20)
    assert [f["geometry"] for f in swapped.factor_info()] == ["sphere", "euclidean", "poincare", "poincare", "sphere"]
    assert split_dims(22, 5) == [5, 5, 4, 4, 4]
    with pytest.raises(ValueError, match="too small"):
        _product("matched", 3)
    with pytest.raises(ValueError, match="not defined"):
        _product("nonsense", 20)


def test_a_concept_is_read_only_from_its_own_factor():
    torch.manual_seed(1)
    m, h = _product("matched", 20)
    x = torch.randn(5, 32)
    before = m(x)["logits"].detach()
    f = m.factor("margin")
    o, d = m.geometry.offsets[f], m.geometry.dims[f]
    with torch.no_grad():  # change only the margin factor's rows of the projection
        m.projector.image_proj.weight[o:o + d] += torch.randn(d, 32)
    after = m(x)["logits"].detach()
    margin_cols = [h.scored_names.index(n) for n in h.scored_names if n.startswith("margin_")]
    other = [i for i in range(len(h.scored_names)) if i not in margin_cols]
    assert torch.allclose(before[:, other], after[:, other])
    assert not torch.allclose(before[:, margin_cols], after[:, margin_cols])


def test_product_optimiser_uses_each_factor_geometry():
    from hypertrace.training import make_optimizer

    m, _ = _product("matched", 20)
    opt, _ = make_optimizer(m, {"lr": 1e-2, "weight_decay": 1e-4, "curvature_lr": 1e-3}, total_steps=10)
    manifold = [g for g in opt.param_groups if g.get("manifold")]
    assert [g["geometry_index"] for g in manifold] == [0, 1, 2, 3, 4]
    assert [opt.geometries[g["geometry_index"]].name for g in manifold] == [
        "euclidean", "poincare", "sphere", "euclidean", "poincare"]
    for _ in range(5):
        opt.zero_grad()
        out = m(torch.randn(8, 32))
        m.loss(out, torch.rand(8, 17))["total"].backward()
        opt.step()
    for f, g in zip(m.factor_points, m.geometry.factors, strict=True):
        assert torch.isfinite(f).all()
        if g.name == "poincare":
            assert f.norm(dim=-1).max() < 1  # still inside the ball
    learned, _ = _product("learned", 20)
    opt, _ = make_optimizer(learned, {"lr": 1e-2, "weight_decay": 1e-4, "curvature_lr": 1e-3}, total_steps=10)
    assert opt.param_groups[-1]["lr"] == 1e-3 and len(opt.param_groups[-1]["params"]) == 5  # one κ per factor
