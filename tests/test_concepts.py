from importlib.resources import files

import numpy as np
import pandas as pd
import pytest
import torch
import yaml

from hypertrace.concepts import ConceptHierarchy, LabelBuilder


@pytest.fixture
def lidc():
    path = files("hypertrace") / "configs/concepts/lidc.yaml"
    return ConceptHierarchy.from_config(yaml.safe_load(path.read_text()))


def test_lidc_tree_loads(lidc):
    assert lidc.root == "chest"
    assert "margin_spiculated" in lidc.scored_names
    assert "nodule" not in lidc.scored_names  # internal nodes are not predicted
    assert lidc.ancestors("margin_spiculated") == ["margin", "nodule", "pulmonary_lesion", "lung", "chest"]
    assert lidc.depth("chest") == 0


def test_edges_and_non_edges_are_disjoint(lidc):
    e = {tuple(x) for x in lidc.edges(transitive=True).tolist()}
    ne = {tuple(x) for x in lidc.non_edges().tolist()}
    assert not e & ne
    assert (lidc.index("nodule"), lidc.index("margin_spiculated")) in e
    assert (lidc.index("margin_spiculated"), lidc.index("nodule")) in ne


def test_invalid_trees_rejected():
    with pytest.raises(ValueError, match="one root"):
        ConceptHierarchy.from_config({"nodes": [{"name": "a", "parent": None}, {"name": "b", "parent": None}]})
    with pytest.raises(ValueError, match="unknown parent"):
        ConceptHierarchy.from_config({"nodes": [{"name": "a", "parent": None}, {"name": "b", "parent": "x"}]})
    with pytest.raises(ValueError, match="No scored"):
        ConceptHierarchy.from_config({"nodes": [{"name": "a", "parent": None}]})


def test_label_builder(lidc):
    df = pd.DataFrame({
        "diameter_mm": [10.0, 5.0],
        "spiculation_r1": [5, 1], "spiculation_r2": [3, np.nan],
        "lobulation_r1": [1, 1], "lobulation_r2": [1, 1],
        "margin_r1": [5, 1], "margin_r2": [5, 1],
        "texture_r1": [5, 1], "texture_r2": [5, 3],
        "calcification_r1": [6, 3], "calcification_r2": [6, 6],
        "sphericity_r1": [5, 2], "sphericity_r2": [4, 3],
    })
    lab = LabelBuilder(lidc).build(df)
    col = {n: i for i, n in enumerate(lab.names)}
    assert lab.soft[0, col["margin_spiculated"]] == pytest.approx(0.75)  # mean(5,3)=4 -> 0.75
    assert lab.soft[1, col["margin_spiculated"]] == pytest.approx(0.0)  # NaN reader ignored
    assert lab.soft[0, col["size_8to15"]] == 1 and lab.soft[0, col["size_lt6"]] == 0  # range bins
    assert lab.soft[1, col["size_lt6"]] == 1 and lab.soft[1, col["size_6to8"]] == 0
    assert lab.soft[0, col["shape_round"]] == 1 and lab.soft[1, col["shape_linear"]] == pytest.approx(0.5)
    assert lab.soft[1, col["texture_subsolid"]] == 1 and lab.soft[1, col["texture_ground_glass"]] == 0.5
    assert lab.soft[1, col["texture_part_solid"]] == 0.5 and lab.soft[0, col["texture_solid"]] == 1
    assert lab.soft[1, col["calcification_present"]] == pytest.approx(0.5)  # 1 of 2 readers
    assert lab.soft[0, col["margin_irregular"]] == pytest.approx(0.0)
    assert lab.binary[0, col["margin_spiculated"]] == 1


def test_tree_v2_structure(lidc):
    assert len(lidc) == 26 and len(lidc.scored_names) == 17
    col = {n: i for i, n in enumerate(lidc.scored_names)}
    assert (col["texture_subsolid"], col["texture_part_solid"]) in lidc.scored_edges()
    groups = [sorted(lidc.scored_names[i] for i in g) for g in lidc.exclusive_groups()]
    assert ["size_6to8", "size_8to15", "size_ge15", "size_lt6"] in groups
    paths = lidc.path_lengths()
    i, j = lidc.index("margin_spiculated"), lidc.index("texture_part_solid")
    assert paths[i, j] == 5 and paths[i, i] == 0 and torch.equal(paths, paths.T)


def test_missing_reader_columns_raise(lidc):
    with pytest.raises(KeyError, match="No reader columns"):
        LabelBuilder(lidc).build(pd.DataFrame({"diameter_mm": [1.0]}))
