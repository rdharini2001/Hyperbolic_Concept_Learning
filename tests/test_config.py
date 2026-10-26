import pytest
import yaml

from hypertrace.config import ConfigError, default_config, load_config, parse_value


def _write(path, data):
    path.write_text(yaml.safe_dump(data))
    return path


def test_packaged_configs_load(monkeypatch):
    monkeypatch.delenv("HT_DATA_ROOT", raising=False)
    cfg = load_config(default_config("train"))
    assert cfg["geometry"]["name"] == "poincare"
    assert cfg["paths"]["manifest"] == "./data/manifest_folds.csv"  # ${HT_DATA_ROOT:-./data} default
    assert cfg["concepts_file"].endswith("configs/labels/lidc.yaml")
    ex = load_config(default_config("extract"))
    assert ex["encoder"]["name"] == "ctclip" and ex["shard_id"] == "0"


def test_base_inheritance_deep_merges(tmp_path):
    _write(tmp_path / "a.yaml", {"x": 1, "m": {"p": 1, "q": 2}, "l": [1, 2]})
    _write(tmp_path / "b.yaml", {"base": "a.yaml", "m": {"q": 3}, "l": [9]})
    cfg = load_config(tmp_path / "b.yaml")
    assert cfg == {"x": 1, "m": {"p": 1, "q": 3}, "l": [9]}


def test_experiment_configs_inherit_train():
    cfg = load_config(default_config("experiments/fmcib"))
    assert cfg["experiment"] == "fmcib" and cfg["data"]["image_feature"] == "lesion_embedding"
    assert cfg["model"]["concepts"]["mode"] == "free" and cfg["model"]["concepts"]["param"] == "riemannian"
    assert cfg["optim"]["lr"] == 5.0e-3  # untouched keys come from train.yaml
    v1 = load_config(default_config("experiments/v1_text_anchored"))
    assert v1["model"]["concepts"]["mode"] == "text" and v1["concepts_file"].endswith("lidc_v1.yaml")


def test_overrides_are_typed_and_strict(tmp_path):
    _write(tmp_path / "c.yaml", {"optim": {"lr": 0.1}, "flag": False, "xs": [1]})
    cfg = load_config(tmp_path / "c.yaml", ["optim.lr=1e-3", "flag=true", "xs=[1, 2]"])
    assert cfg["optim"]["lr"] == 1e-3 and cfg["flag"] is True and cfg["xs"] == [1, 2]
    with pytest.raises(ConfigError, match="unknown key"):
        load_config(tmp_path / "c.yaml", ["optim.lrr=1"])  # typo must not pass silently
    with pytest.raises(ConfigError, match="not a section"):
        load_config(tmp_path / "c.yaml", ["flag.x=1"])


def test_env_expansion(tmp_path, monkeypatch):
    _write(tmp_path / "e.yaml", {"a": "${HT_TEST_VAR}/x", "b": "${HT_UNSET_VAR:-fallback}", "c": "${HT_NOPE}"})
    monkeypatch.setenv("HT_TEST_VAR", "/scratch")
    monkeypatch.delenv("HT_NOPE", raising=False)
    with pytest.raises(ConfigError, match="HT_NOPE"):
        load_config(tmp_path / "e.yaml")
    monkeypatch.setenv("HT_NOPE", "ok")
    cfg = load_config(tmp_path / "e.yaml")
    assert cfg == {"a": "/scratch/x", "b": "fallback", "c": "ok"}


def test_circular_base_rejected(tmp_path):
    _write(tmp_path / "a.yaml", {"base": "b.yaml"})
    _write(tmp_path / "b.yaml", {"base": "a.yaml"})
    with pytest.raises(ConfigError, match="Circular"):
        load_config(tmp_path / "a.yaml")


@pytest.mark.parametrize("text,value", [("1e-3", 1e-3), ("5", 5), ("true", True), ("null", None), ("abc", "abc"),
                                        ("[0.1, 10]", [0.1, 10])])
def test_parse_value(text, value):
    assert parse_value(text) == value
