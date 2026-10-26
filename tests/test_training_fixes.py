"""Unit tests for the training-fix patch: per-geometry clip, head temperature, input dropout, PCA, trials."""

import pytest
import torch

from hypertrace.cli.tune import draw_trials
from hypertrace.geometry import build_geometry
from hypertrace.models.concept_head import DistanceConceptHead
from hypertrace.models.projector import ConceptSpaceProjector, resolve_clip_r
from hypertrace.training import selection_sign


def test_clip_r_per_geometry():
    assert resolve_clip_r({"poincare": 2.0, "euclidean": None}, "poincare") == 2.0
    assert resolve_clip_r({"poincare": 2.0, "euclidean": None}, "euclidean") is None
    assert resolve_clip_r(3, "euclidean") == 3.0
    with pytest.raises(ValueError, match="no entry"):
        resolve_clip_r({"poincare": 2.0}, "euclidean")


def test_euclidean_projector_without_cap_reaches_large_norms():
    torch.manual_seed(0)
    g = build_geometry("euclidean")
    p = ConceptSpaceProjector(g, image_dim=16, text_dim=None, embed_dim=4, clip_r={"poincare": 2.0, "euclidean": None},
                              unit_norm_inputs=False)
    with torch.no_grad():
        p.log_alpha_image.fill_(3.0)
    assert p.embed_image(torch.randn(32, 16)).norm(dim=-1).max() > 2.0  # no ring at the cap
    pc = ConceptSpaceProjector(build_geometry("poincare"), image_dim=16, text_dim=None, embed_dim=4,
                               clip_r={"poincare": 2.0, "euclidean": None}, unit_norm_inputs=False)
    with torch.no_grad():
        pc.log_alpha_image.fill_(3.0)
    assert pc.embed_image(torch.randn(32, 16)).norm(dim=-1).max() < 1  # inside the ball


def test_input_dropout_only_in_training():
    torch.manual_seed(0)
    p = ConceptSpaceProjector(build_geometry("euclidean"), image_dim=64, text_dim=None, embed_dim=4,
                              unit_norm_inputs=False, input_dropout=0.5, clip_r=None)
    x = torch.randn(8, 64)
    p.eval()
    assert torch.equal(p.embed_image(x), p.embed_image(x))
    p.train()
    assert not torch.allclose(p.embed_image(x), p.embed_image(x))


def test_per_concept_tau():
    h = DistanceConceptHead(build_geometry("euclidean"), n_concepts=5, init_tau=0.2, per_concept_tau=True)
    assert h.log_tau.shape == (5,)
    logits = h(torch.randn(3, 4), torch.randn(5, 4))
    assert logits.shape == (3, 5)
    assert DistanceConceptHead(build_geometry("euclidean"), 5).log_tau.shape == ()


def test_selection_direction():
    assert selection_sign("val/loss_concept") == 1 and selection_sign("val/loss") == 1
    assert selection_sign("val/auroc_macro") == -1 and selection_sign("val/brier_macro") == 1


def test_trials_are_deterministic_and_start_with_base():
    space = {"sample": "random", "n_trials": 4, "seed": 3, "params": {"a": [1, 2, 3], "b": [True, False]}}
    t1, t2 = draw_trials(space), draw_trials(space)
    assert t1 == t2 and t1[0] == {} and len(t1) == 4 and len({tuple(sorted(t.items())) for t in t1}) == 4
    assert len(draw_trials({**space, "sample": "grid"})) == 1 + 6
