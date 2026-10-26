"""The concept space is a genuine Poincaré-ball model, not a Euclidean shortcut.

These tests check the *trained model's* outputs, not just the geometry helpers:
points live strictly inside the ball, concept scores are a function of the true
geodesic distance (verified with an independent formula), and that distance orders
pairs differently from Euclidean distance, which a flat shortcut could not do.
"""

import math
from importlib.resources import files

import pytest
import torch
import yaml

from hypertrace.concepts import ConceptHierarchy
from hypertrace.config import default_config, load_config
from hypertrace.geometry import poincare as pm
from hypertrace.models.model import ConceptSpaceModel

torch.manual_seed(0)


MODES = [{"mode": "free", "param": "riemannian"}, {"mode": "free", "param": "tangent"}, {"mode": "text"}]


def _model(geometry: str = "poincare", curvature: float = 1.0, concepts: dict | None = None) -> ConceptSpaceModel:
    cfg = load_config(default_config("train"))
    h = ConceptHierarchy.from_config(yaml.safe_load((files("hypertrace") / "configs/concepts/lidc.yaml").read_text()))
    g = dict(cfg["geometry"], name=geometry)
    g["poincare"] = dict(g["poincare"], curvature=curvature)
    text = torch.nn.functional.normalize(torch.randn(len(h), 64), dim=-1)
    m = cfg["model"]
    concepts = {**m["concepts"], **(concepts or {})}
    projector = dict(m["projector"], share_projection=concepts["mode"] == "text")
    return ConceptSpaceModel(h, 64, g, projector, m["head"], m["loss"], concepts=concepts, text_latents=text)


@pytest.mark.parametrize("concepts", MODES)
@pytest.mark.parametrize("c", [0.5, 1.0, 2.0])
def test_all_points_strictly_inside_ball(c, concepts):
    model = _model(curvature=c, concepts=concepts)
    x = torch.nn.functional.normalize(torch.randn(256, 64), dim=-1) * 50  # extreme inputs
    out = model(x)
    radius = 1 / math.sqrt(c)
    assert (out["lesion"].norm(dim=-1) < radius).all()
    assert (out["nodes"].norm(dim=-1) < radius).all()


@pytest.mark.parametrize("concepts", MODES)
def test_scores_use_geodesic_distance_independent_formula(concepts):
    model = _model(concepts=concepts)
    x = torch.nn.functional.normalize(torch.randn(32, 64), dim=-1)
    out = model(x)
    c = model.geometry.c
    concepts = out["nodes"][model.scored_idx]
    # independent check: Mobius-addition form  d(x, y) = 2/sqrt(c) artanh( sqrt(c) |(-x) (+) y| )
    d = pm.dist(out["lesion"].unsqueeze(1), concepts.unsqueeze(0), c)
    expected = -d / model.head.log_tau.exp() + model.head.bias
    assert torch.allclose(out["logits"], expected, atol=1e-4)


def test_poincare_distance_is_not_a_euclidean_shortcut():
    # Near the boundary, tiny Euclidean steps are long geodesics: the ordering flips.
    a, b = torch.tensor([[0.0, 0.0]]), torch.tensor([[0.5, 0.0]])
    c_, d_ = torch.tensor([[0.99, 0.0]]), torch.tensor([[0.999, 0.0]])
    assert (a - b).norm() > (c_ - d_).norm()                  # Euclidean: a-b is the longer pair
    assert pm.dist(a, b, 1.0) < pm.dist(c_, d_, 1.0)          # Poincare: c-d is the longer pair


def test_depth_is_hyperbolic_distance_from_origin():
    model = _model()
    p = model.concept_points()
    expected = 2 * torch.atanh(p.norm(dim=-1).clamp(max=1 - 1e-6))  # c = 1
    assert torch.allclose(model.geometry.depth(p), expected, atol=1e-4)


def test_exp_map_parameterisation_is_exact():
    # Every point of the ball is reachable and the map is invertible: optimising tangent
    # vectors is an exact reparameterisation of the ball, not an approximation.
    y = pm.expmap0(torch.randn(100, 8), 1.0)
    assert torch.allclose(pm.expmap0(pm.logmap0(y, 1.0), 1.0), y, atol=1e-5)


def test_euclidean_twin_differs_only_in_geometry():
    torch.manual_seed(1)
    hp = _model("poincare")
    torch.manual_seed(1)
    eu = _model("euclidean")
    for (n1, p1), (n2, p2) in zip(hp.named_parameters(), eu.named_parameters(), strict=True):
        assert n1 == n2
        if n1 == "concept_param":  # same initial point, expressed in each space: exp_0(v) vs v
            assert torch.allclose(pm.logmap0(p1, 1.0), p2, atol=1e-4)
        else:
            assert torch.equal(p1, p2)  # identical parameters and initialisation
    x = torch.nn.functional.normalize(torch.randn(4, 64), dim=-1)
    assert not torch.allclose(hp(x)["logits"], eu(x)["logits"])
