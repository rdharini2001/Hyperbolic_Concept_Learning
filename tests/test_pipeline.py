"""End-to-end on a synthetic copy of the bucket: CSVs -> manifest -> folds -> scan extraction -> training."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
import yaml

from hypertrace import training
from hypertrace.cli import build_manifest, extract, make_folds, train, zeroshot
from hypertrace.config import default_config
from hypertrace.data.folds import assert_no_patient_leakage, split_indices
from hypertrace.data.synthetic import make_synthetic_bucket

FMCIB = default_config("experiments/fmcib")
V1 = default_config("experiments/v1_text_anchored")
TANGENT = default_config("experiments/tangent_concepts")


def _prepare(root: Path, n_patients: int = 30, n_folds: int = 5) -> Path:
    bucket = make_synthetic_bucket(root / "bucket", n_patients=n_patients)
    build_manifest.main(["--bucket-dir", str(bucket), "--out", str(root / "data" / "manifest.csv")])
    make_folds.main(["--manifest", str(root / "data" / "manifest.csv"),
                     "--out", str(root / "data" / "manifest_folds.csv"), "--n-folds", str(n_folds)])
    return root


def _extract(root: Path, *extra: str) -> None:
    extract.main(["--set", f"paths.bucket_root={root / 'bucket'}",
                  f"paths.manifest={root / 'data' / 'manifest_folds.csv'}", f"paths.cache_root={root / 'cache'}",
                  "encoder.name=fake", "shard_id=0", "num_shards=1", "device=cpu", *extra])


def _train_args(root: Path, *extra: str, config: Path | None = None, run_dir: Path | None = None) -> list[str]:
    head = (["--config", str(config)] if config else []) + (["--run-dir", str(run_dir)] if run_dir else [])
    return head + ["--set", f"paths.manifest={root / 'data' / 'manifest_folds.csv'}",
                   f"paths.cache_dir={root / 'cache' / 'fake_scan_dil0'}", f"paths.output_root={root / 'out'}",
                   "trainer.device=cpu", *extra]


@pytest.fixture(scope="module")
def workspace(tmp_path_factory):
    pytest.importorskip("nibabel")
    root = _prepare(tmp_path_factory.mktemp("ht"))
    _extract(root)
    return root


def test_folds_are_patient_disjoint(workspace):
    df = pd.read_csv(workspace / "data" / "manifest_folds.csv")
    for k in range(5):
        assert_no_patient_leakage(df, split_indices(df["fold"].to_numpy(), k, 5))


def test_extraction_is_idempotent(workspace):
    shards = sorted((workspace / "cache" / "fake_scan_dil0" / "shards").glob("*.pt"))
    mtimes = [s.stat().st_mtime for s in shards]
    _extract(workspace)
    assert [s.stat().st_mtime for s in shards] == mtimes  # nothing re-encoded


def test_cache_rejects_changed_settings(workspace):
    with pytest.raises(RuntimeError, match="different settings"):
        _extract(workspace, f"cache_dir={workspace / 'cache' / 'fake_scan_dil0'}", "hu_window=[-500.0, 500.0]")


def test_extract_saves_config_with_metadata(workspace):
    cfg = yaml.safe_load((workspace / "cache" / "fake_scan_dil0" / "extract_logs" / "config_shard0.yaml").read_text())
    assert cfg["encoder"]["name"] == "fake" and "torch" in cfg["_meta"] and cfg["concepts"]["nodes"]


@pytest.mark.parametrize("geometry", ["poincare", "euclidean"])
def test_train_end_to_end(workspace, geometry):
    test_metrics = train.run(_train_args(workspace, f"geometry.name={geometry}", "trainer.max_epochs=3",
                                         "data.batch_size=16"))
    run_dir = workspace / "out" / "runs" / f"main_{geometry}_token_mean_d64_f0_s0"
    assert (run_dir / "checkpoints" / "last.pt").exists() and (run_dir / "checkpoints" / "best.pt").exists()
    preds = pd.read_csv(run_dir / "predictions_test.csv")
    assert preds["fold"].nunique() == 1 and preds["fold"].iloc[0] == 0  # out-of-fold only
    assert preds.filter(like="prob_").shape[1] == 17  # scored concepts: configs/labels/lidc.yaml (v3)
    probs = preds.filter(like="prob_").to_numpy()
    assert ((probs >= 0) & (probs <= 1)).all()
    space = torch.load(run_dir / "concept_space.pt", weights_only=False)
    assert space["geometry"] == geometry and torch.isfinite(space["points"]).all()
    assert "test/auroc_macro" in test_metrics and "test/edge_in_cone" in test_metrics
    for k in ("tree_map", "dist_spearman", "parent_child_violation", "exclusive_violation"):
        assert f"test/{k}" in test_metrics, k
    m = json.loads((run_dir / "metrics.json").read_text())
    assert "init/tree_map" in m["init"]
    assert (run_dir / "feature_norm.pt").exists()  # data.standardize=true by default
    assert m["epochs_run"] == 3 and len(pd.read_csv(run_dir / "metrics_per_epoch.csv")) == 3
    saved = yaml.safe_load((run_dir / "config.yaml").read_text())
    assert saved["geometry"]["name"] == geometry and saved["_meta"]["command"]


def test_console_entry_returns_none(workspace):
    # console scripts call sys.exit(main()); anything but None would exit with status 1 on SLURM
    assert train.main(_train_args(workspace, "trainer.max_epochs=1", "seed=11")) is None


def test_resume_matches_uninterrupted_run(workspace, tmp_path, monkeypatch):
    """A job pre-empted after 2 of 4 epochs and requeued must end exactly where an uninterrupted run ends."""
    common = ("seed=7", "trainer.early_stopping_patience=1000", "trainer.max_epochs=4")
    full, split = tmp_path / "full", tmp_path / "split"
    train.run(_train_args(workspace, *common, run_dir=full))

    orig, calls = training.train_one_epoch, {"n": 0}

    def preempt_after_two(*a, **k):
        if calls["n"] == 2:
            raise KeyboardInterrupt  # SLURM pre-emption
        calls["n"] += 1
        return orig(*a, **k)

    monkeypatch.setattr(training, "train_one_epoch", preempt_after_two)
    with pytest.raises(KeyboardInterrupt):
        train.run(_train_args(workspace, *common, run_dir=split))
    monkeypatch.setattr(training, "train_one_epoch", orig)
    train.run(_train_args(workspace, *common, run_dir=split))  # requeued job

    ref = torch.load(full / "checkpoints" / "last.pt", weights_only=False)
    got = torch.load(split / "checkpoints" / "last.pt", weights_only=False)
    assert got["epoch"] == ref["epoch"] == 3
    for k in ref["model"]:
        assert torch.allclose(got["model"][k], ref["model"][k], atol=1e-6), k
    assert len(pd.read_csv(split / "metrics_per_epoch.csv")) == 4  # history kept across the restart


def test_concepts_move_on_the_ball_with_riemannian_adam(workspace, tmp_path):
    train.run(_train_args(workspace, "trainer.max_epochs=3", "seed=3", run_dir=tmp_path / "r"))
    ck = torch.load(tmp_path / "r" / "checkpoints" / "last.pt", weights_only=False)
    pts = ck["model"]["concept_param"]
    assert pts.norm(dim=-1).max() < 1  # points stayed inside the ball
    groups = ck["optimizer"]["param_groups"]
    assert [g["manifold"] for g in groups] == [False, False, True]


def test_share_projection_needs_text_mode(workspace):
    with pytest.raises(ValueError, match="share_projection"):
        train.run(_train_args(workspace, "model.projector.share_projection=true"))


def test_experiment_configs_inherit_and_rename(workspace):
    train.run(_train_args(workspace, "trainer.max_epochs=1", config=TANGENT))
    assert (workspace / "out" / "runs" / "tangent_poincare_token_mean_d64_f0_s0" / "metrics.json").exists()
    train.run(_train_args(workspace, "trainer.max_epochs=1", "model.projector.embed_dim=4"))
    assert (workspace / "out" / "runs" / "main_poincare_token_mean_d4_f0_s0" / "metrics.json").exists()


def test_v1_text_anchored_still_runs(workspace, tmp_path):
    import shutil
    cache = tmp_path / "v1cache"
    shutil.copytree(workspace / "cache" / "fake_scan_dil0", cache)
    v1_tree = V1.parent.parent / "concepts" / "lidc_v1.yaml"
    _extract(workspace, f"cache_dir={cache}", "only_text=true", f"concepts_file={v1_tree}")
    m = train.run(_train_args(workspace, "trainer.max_epochs=1", f"paths.cache_dir={cache}", config=V1))
    assert "test/auroc_macro" in m
    with pytest.raises(ValueError, match="different concept tree"):  # v2 tree with a v1 text.pt
        train.run(_train_args(workspace, "trainer.max_epochs=1", f"paths.cache_dir={cache}",
                              "model.concepts.init=text", run_dir=tmp_path / "v2_text_init"))


def test_one_readout_per_manifest_nodule(workspace):
    df = pd.read_csv(workspace / "data" / "manifest_folds.csv")
    shards = list((workspace / "cache" / "fake_scan_dil0" / "shards").glob("*.pt"))
    assert len(shards) == len(df)
    assert min(torch.load(s)["occupancy_sum"].item() for s in shards) > 0  # every nodule inside its window


def test_crop_mode_extraction(workspace):
    _extract(workspace, "input_mode=crop", "crop_size_mm=32")
    assert len(list((workspace / "cache" / "fake_crop_dil0" / "shards").glob("*.pt"))) == \
        len(pd.read_csv(workspace / "data" / "manifest_folds.csv"))


def test_wrong_instance_label_is_caught(tmp_path):
    pytest.importorskip("nibabel")
    root = _prepare(tmp_path, n_patients=10, n_folds=2)
    m = pd.read_csv(root / "data" / "manifest_folds.csv")
    m["consensus_voxels"] = (m["consensus_voxels"] * 3).astype(np.int64)  # simulate a label-id mismatch
    m.to_csv(root / "data" / "manifest_folds.csv", index=False)
    with pytest.raises(extract.InstanceLabelError, match="consensus_voxels"):
        _extract(root)


def test_zeroshot_probe_runs(workspace):
    out = workspace / "zs.csv"
    zeroshot.main(["--cache-dir", str(workspace / "cache" / "fake_scan_dil0"),
                   "--manifest", str(workspace / "data" / "manifest_folds.csv"), "--probe", "--folds", "3",
                   "--out", str(out)])
    res = pd.read_csv(out)
    assert {"zeroshot", "probe"} <= set(res.columns)
    assert res.loc[res.feature == "token_mean", "zeroshot"].isna().all()  # not in CLIP space
    assert res.loc[res.feature == "masked_latent", "zeroshot"].notna().any()


def test_visualize_both_geometries(workspace, tmp_path):
    pytest.importorskip("matplotlib")
    from hypertrace.cli import visualize

    runs = []
    for g in ("poincare", "euclidean"):
        train.run(_train_args(workspace, f"geometry.name={g}", "trainer.max_epochs=2", "seed=21"))
        runs += ["--run", str(workspace / "out" / "runs" / f"main_{g}_token_mean_d64_f0_s21")]
    visualize.main(runs + ["--out", str(tmp_path), "--no-html", "--color-by", "malignancy"])
    assert (tmp_path / "concept_space.png").stat().st_size > 10_000
    coords = pd.read_csv(tmp_path / "main_poincare_token_mean_d64_f0_s21_coords.csv")
    concepts, nodules = coords[coords.kind == "concept"], coords[coords.kind == "nodule"]
    assert len(concepts) == 50 and len(nodules) == len(pd.read_csv(workspace / "data" / "manifest_folds.csv"))
    # radial layout: distance from the centre IS the depth (exact, not approximated)
    r = np.hypot(coords.radial_x, coords.radial_y)
    assert np.allclose(r, coords.depth, atol=1e-5)
    assert (np.hypot(nodules.disk_x, nodules.disk_y) < 1).all()  # inside the Poincaré disk (c = 1)


def test_wrong_cache_dir_says_where_it_looked(workspace, tmp_path):
    df = pd.read_csv(workspace / "data" / "manifest_folds.csv")
    from hypertrace.data.embedding_cache import EmbeddingCache
    with pytest.raises(FileNotFoundError, match="not an embedding cache"):
        EmbeddingCache(tmp_path / "nowhere").load_images(df["nodule_id"].tolist())


def test_lesion_encoder_extraction_and_training(workspace, tmp_path):
    _extract(workspace, "encoder.name=fake_lesion")
    cache = workspace / "cache" / "fake_lesion_lesion"
    shards = list((cache / "shards").glob("*.pt"))
    assert len(shards) == len(pd.read_csv(workspace / "data" / "manifest_folds.csv"))
    s = torch.load(shards[0])
    assert set(s) == {"lesion_embedding", "occupancy_sum"} and s["lesion_embedding"].shape == (48,)
    assert not (cache / "text.pt").exists()  # no text encoder
    m = train.run(_train_args(workspace, "trainer.max_epochs=2", f"paths.cache_dir={cache}", config=FMCIB))
    assert "test/auroc_macro" in m
    with pytest.raises(ValueError, match="no text.pt"):
        train.run(_train_args(workspace, "trainer.max_epochs=1", f"paths.cache_dir={cache}",
                              "model.concepts.init=text", config=FMCIB, run_dir=tmp_path / "text_init"))


def test_radiomics_and_probe_baselines(workspace):
    from hypertrace.cli import radiomics

    rad = workspace / "radiomics.csv"
    radiomics.main(["--manifest", str(workspace / "data" / "manifest_folds.csv"),
                    "--bucket-dir", str(workspace / "bucket"), "--out", str(rad)])
    feats = pd.read_csv(rad)
    assert len(feats) == len(pd.read_csv(workspace / "data" / "manifest_folds.csv"))
    assert feats.shape[1] == 22  # nodule_id + 21 features
    assert feats["sphericity"].between(0, 1.2).all() and (feats["volume_mm3"] > 0).all()
    out = workspace / "zs_rad.csv"
    zeroshot.main(["--cache-dir", str(workspace / "cache" / "fake_scan_dil0"),
                   "--manifest", str(workspace / "data" / "manifest_folds.csv"),
                   "--radiomics", str(rad), "--folds", "3", "--out", str(out)])
    res = pd.read_csv(out)
    assert {"diameter_only", "radiomics", "token_mean"} <= set(res.feature)
    assert res.loc[res.feature == "diameter_only", "probe"].notna().any()


# ------------------------------------------------------------------ training-fix patch
def test_select_metric_and_train_curves(workspace, tmp_path):
    common = ("trainer.max_epochs=6", "trainer.early_stopping_patience=1000", "seed=5", "trainer.select_from_epoch=0")
    for metric in ("val/loss_concept", "val/auroc_macro"):
        rd = tmp_path / metric.replace("/", "_")
        train.run(_train_args(workspace, *common, f"trainer.select_metric={metric}", run_dir=rd))
        curves = pd.read_csv(rd / "metrics_per_epoch.csv")
        assert {"train/auroc_macro", "val/loss_concept", "val/loss_hierarchy"} <= set(curves.columns)
        col = curves[metric]
        expected = int(col.idxmin() if metric.endswith("loss_concept") else col.idxmax())  # first best epoch
        m = json.loads((rd / "metrics.json").read_text())
        assert m["select_metric"] == metric and m["best_epoch"] == expected
        assert "train/auroc_macro" in m["train"]
        best = torch.load(rd / "checkpoints" / "best.pt", weights_only=False)
        assert best["select_metric"] == metric and best["select_value"] == pytest.approx(col.iloc[expected])


def test_no_selection_before_select_from_epoch(workspace, tmp_path):
    rd = tmp_path / "warm"
    train.run(_train_args(workspace, "trainer.max_epochs=6", "trainer.early_stopping_patience=1",
                          "trainer.select_from_epoch=3", "seed=6", run_dir=rd))
    m = json.loads((rd / "metrics.json").read_text())
    assert m["best_epoch"] >= 3 and m["epochs_run"] >= 4  # no early stop during the warm-up


def test_selecting_on_test_is_refused(workspace, tmp_path):
    with pytest.raises(ValueError, match="validation metric"):
        train.run(_train_args(workspace, "trainer.select_metric=test/auroc_macro", run_dir=tmp_path / "r"))


def test_probe_diagnostics_and_input_probe_cache(workspace, tmp_path):
    out = tmp_path / "out"
    args = ("trainer.max_epochs=2", f"paths.output_root={out}")
    m1 = train.run(_train_args(workspace, *args, "seed=1"))
    for k in ("test/probe_embedding/auroc_macro", "test/probe_input/auroc_macro"):
        assert 0.0 <= m1[k] <= 1.0, k
    saved = json.loads((out / "runs" / "main_poincare_token_mean_d64_f0_s1" / "metrics.json").read_text())
    assert 0.0 <= saved["val"]["val/probe_input/auroc_macro"] <= 1.0
    refs = list((out / "reference").glob("input_probe_*.json"))
    assert len(refs) == 1  # one per fold, shared by every run of that fold
    m2 = train.run(_train_args(workspace, *args, "seed=2", "geometry.name=euclidean"))
    assert m2["test/probe_input/auroc_macro"] == m1["test/probe_input/auroc_macro"]  # reused, not refitted
    assert len(list((out / "reference").glob("input_probe_*.json"))) == 1
    train.run(_train_args(workspace, *args, "seed=1", "data.test_fold=1"))
    assert len(list((out / "reference").glob("input_probe_*.json"))) == 2  # new fold, new reference


def test_run_folder_with_other_settings_is_refused_and_tag_separates(workspace, tmp_path):
    from hypertrace.config import ConfigError

    out = tmp_path / "out"
    base = ("trainer.max_epochs=1", f"paths.output_root={out}")
    train.run(_train_args(workspace, *base))
    train.run(_train_args(workspace, *base))  # same settings: resumes / finishes instantly
    with pytest.raises(ConfigError, match="different settings.*optim.weight_decay"):
        train.run(_train_args(workspace, *base, "optim.weight_decay=0.1"))
    train.run(_train_args(workspace, *base, "optim.weight_decay=0.1", "tag=wd"))
    assert (out / "runs" / "main_poincare_token_mean_d64_twd_f0_s0" / "metrics.json").exists()


def test_euclidean_lesions_are_not_capped_by_default(workspace, tmp_path):
    from hypertrace.viz import load_run_model

    rd = tmp_path / "e"
    train.run(_train_args(workspace, "geometry.name=euclidean", "trainer.max_epochs=1",
                          "model.projector.input_dropout=0.3", "data.pca_dim=8", run_dir=rd))
    _, model, data = load_run_model(rd)
    assert model.projector.clip_r is None and data.x.shape[1] == 8
    norm = torch.load(rd / "feature_norm.pt", weights_only=False)
    assert norm["pca_components"].shape[0] == 8 and norm["pca_components"].shape[1] > 8


def test_tune_runs_matched_trials_and_selects_on_val(workspace, tmp_path):
    from hypertrace.cli import tune

    space = tmp_path / "space.yaml"
    space.write_text(yaml.safe_dump({"name": "t", "sample": "random", "n_trials": 2, "seed": 0,
                                     "params": {"optim.weight_decay": [0.01, 0.1]}}))
    out = tmp_path / "tune"
    common = ["--space", str(space), "--dims", "4", "--folds", "0", "1", "--seeds", "0", "--out", str(out),
              "--set", f"paths.manifest={workspace / 'data' / 'manifest_folds.csv'}",
              f"paths.cache_dir={workspace / 'cache' / 'fake_scan_dil0'}", f"paths.output_root={tmp_path / 'o'}",
              "trainer.device=cpu", "trainer.max_epochs=2"]
    tune.main(common)
    runs = pd.read_csv(out / "runs.csv")
    assert len(runs) == 2 * 2 * 2  # 2 trials x 2 geometries x 2 folds
    # equal budget: every geometry ran exactly the same (trial, fold) pairs
    pairs = {g: set(map(tuple, d[["trial", "fold"]].to_numpy())) for g, d in runs.groupby("geometry")}
    assert pairs["poincare"] == pairs["euclidean"]
    sel = pd.read_csv(out / "selected_per_fold.csv")
    for (g, f), d in runs.groupby(["geometry", "fold"]):
        chosen = sel[(sel.geometry == g) & (sel.fold == f)].trial.item()
        assert chosen == d.loc[d.val.idxmax(), "trial"]  # chosen on validation, not test
    trials = pd.read_csv(out / "trials.csv")
    assert trials.loc[0].drop(["trial", "tag"]).isna().all()  # trial 0 = base config


def test_concept_groups_and_legend(workspace, tmp_path):
    from hypertrace.cli.visualize import concept_styles, legend_columns
    from hypertrace.viz import build_view

    rd = tmp_path / "v"
    train.run(_train_args(workspace, "trainer.max_epochs=1", run_dir=rd))
    v = build_view(rd)
    groups = dict(zip(v.node_names, v.node_group, strict=True))
    assert groups["chest"] is None and groups["nodule"] is None  # trunk
    assert groups["size_lt6"] == "size" and groups["type_part_solid"] == "nodule_type"
    assert groups["solid_component_ge8"] == "nodule_type" and groups["perifissural"] == "location"
    styles = concept_styles(v)
    scored = [(s.color, s.marker) for s in styles if s.scored]
    assert len(set(scored)) == len(scored)  # every scored concept has its own colour + marker
    handles, labels, headers = legend_columns(styles)
    assert {labels[i] for i in headers} == {"Size", "Nodule type", "Shape", "Margin", "Calcification",
                                            "Internal features", "Location", "Tree"}
    assert "< 6 mm" in labels and "spiculated" in labels  # clinical display terms from the ontology
    assert "perifissural" in labels and "hollow = unscored" in labels


def test_v2_tree_experiment_still_runs(workspace, tmp_path):
    m = train.run(_train_args(workspace, "trainer.max_epochs=1", config=default_config("experiments/tree_v2"),
                              run_dir=tmp_path / "v2"))
    assert "test/auroc_macro" in m
    saved = yaml.safe_load((tmp_path / "v2" / "config.yaml").read_text())
    assert len(saved["concepts"]["nodes"]) == 26


def test_v3_labels_equal_v2_labels(workspace):
    """The 17 scored v3 concepts carry exactly the v2 labels (only names changed)."""
    from hypertrace.concepts import ConceptHierarchy, LabelBuilder
    from hypertrace.concepts.ontology import load_concept_tree

    df = pd.read_csv(workspace / "data" / "manifest_folds.csv")
    v2 = LabelBuilder(ConceptHierarchy.from_config(load_concept_tree(default_config("concepts/lidc")))).build(df)
    v3 = LabelBuilder(ConceptHierarchy.from_config(load_concept_tree(default_config("labels/lidc")))).build(df)
    rename = {"texture_solid": "type_solid", "texture_subsolid": "type_subsolid",
              "texture_part_solid": "type_part_solid", "texture_ground_glass": "type_ground_glass",
              "shape_ovoid": "shape_oval"}
    assert sorted(rename.get(n, n) for n in v2.names) == sorted(v3.names)
    for i, n in enumerate(v2.names):
        j = v3.names.index(rename.get(n, n))
        assert np.array_equal(v2.soft[:, i], v3.soft[:, j], equal_nan=True), n


# ------------------------------------------------------------------ step 3: sphere, learned κ, product
@pytest.mark.parametrize("geometry", ["sphere", "kstereo"])
def test_single_space_sphere_and_learned_curvature(workspace, tmp_path, geometry):
    rd = tmp_path / geometry
    m = train.run(_train_args(workspace, f"geometry.name={geometry}", "trainer.max_epochs=3", run_dir=rd))
    assert 0.0 <= m["test/auroc_macro"] <= 1.0 and "test/group/size/ordinal_spearman" in m
    curves = pd.read_csv(rd / "metrics_per_epoch.csv")
    if geometry == "kstereo":
        assert curves["train/curvature"].iloc[-1] != 0.0  # κ is learned (started at 0)
    space = torch.load(rd / "concept_space.pt", weights_only=False)
    assert space["geometry"] == geometry and torch.isfinite(space["points"]).all()


def test_product_end_to_end_with_group_metrics(workspace, tmp_path):
    out = tmp_path / "out"
    for assignment in ("matched", "swapped"):
        m = train.run(_train_args(workspace, "geometry.name=product", f"geometry.product.assignment={assignment}",
                                  "model.projector.embed_dim=20", "trainer.max_epochs=3", f"paths.output_root={out}"))
        for k in ("test/group/size/ordinal_spearman", "test/group/size/ordinal_dist_spearman",
                  "test/group/nodule_type/parent_child_violation", "test/group/shape/exclusive_violation",
                  "test/group/nodule_type/edge_in_cone", "test/nodule_type/tree_map", "test/tree_map",
                  "test/probe_embedding/auroc_macro"):
            assert k in m, (assignment, k)
    rd = out / "runs" / "main_product-matched_token_mean_d20_f0_s0"
    space = torch.load(rd / "concept_space.pt", weights_only=False)
    assert [f["geometry"] for f in space["factors"]] == ["euclidean", "poincare", "sphere", "euclidean", "poincare"]
    curves = pd.read_csv(rd / "metrics_per_epoch.csv")
    assert "train/curvature/nodule_type" in curves.columns and curves["train/curvature/nodule_type"].iloc[0] == -1.0

    from hypertrace.cli import summarize, visualize
    visualize.main(["--run", str(rd), "--out", str(tmp_path / "figs"), "--no-html"])
    rows = list((tmp_path / "figs").glob("*_coords.csv"))
    assert len(rows) == 5  # one view per factor
    shape = pd.read_csv(next(r for r in rows if "__shape" in r.name))
    assert set(shape[shape.kind == "concept"].group) == {"shape"}
    summarize.main(["--runs", str(out / "runs"), "--groups"])


def test_product_resume_matches_uninterrupted_run(workspace, tmp_path, monkeypatch):
    common = ("seed=7", "trainer.early_stopping_patience=1000", "trainer.max_epochs=4", "geometry.name=product",
              "geometry.product.assignment=learned", "model.projector.embed_dim=20")
    full, split = tmp_path / "full", tmp_path / "split"
    train.run(_train_args(workspace, *common, run_dir=full))
    orig, calls = training.train_one_epoch, {"n": 0}

    def preempt_after_two(*a, **k):
        if calls["n"] == 2:
            raise KeyboardInterrupt
        calls["n"] += 1
        return orig(*a, **k)

    monkeypatch.setattr(training, "train_one_epoch", preempt_after_two)
    with pytest.raises(KeyboardInterrupt):
        train.run(_train_args(workspace, *common, run_dir=split))
    monkeypatch.setattr(training, "train_one_epoch", orig)
    train.run(_train_args(workspace, *common, run_dir=split))
    ref = torch.load(full / "checkpoints" / "last.pt", weights_only=False)
    got = torch.load(split / "checkpoints" / "last.pt", weights_only=False)
    for k in ref["model"]:
        assert torch.allclose(got["model"][k], ref["model"][k], atol=1e-6), k
