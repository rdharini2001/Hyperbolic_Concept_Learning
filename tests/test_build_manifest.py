from importlib.resources import files

import numpy as np
import pandas as pd
import pytest
import yaml

from hypertrace.cli.build_manifest import DEFAULT_MAP, MappingError, build_manifest, main, parse_term
from hypertrace.concepts import ConceptHierarchy, LabelBuilder
from hypertrace.data.manifest import load_manifest
from hypertrace.data.synthetic import make_synthetic_bucket

CFG = yaml.safe_load(DEFAULT_MAP.read_text())


@pytest.fixture(scope="module")
def bucket(tmp_path_factory):
    pytest.importorskip("nibabel")
    return make_synthetic_bucket(tmp_path_factory.mktemp("bucket"), n_patients=12, shape=(16, 24, 24))


def test_parse_term_named_and_scaled():
    s = pd.Series(["Circumscribed margin", "3 out of 5", "Indistinct margin", np.nan,
                   "5 out of 5 (Marked spiculation)"])
    out = parse_term(s, "margin", CFG["annotations"]["value_maps"]["margin"])
    assert out.tolist()[:3] == [5, 3, 1] and np.isnan(out[3]) and out[4] == 5
    with pytest.raises(MappingError, match="unrecognised"):
        parse_term(pd.Series(["Spiky"]), "margin", {})


def test_manifest_from_bucket_schema(bucket):
    df = build_manifest(bucket, CFG)
    nod = pd.read_csv(bucket / "nodules.csv")
    assert len(df) == int(nod["in_default_gt"].sum())  # default GT: >= 3 annotations
    assert df["nodule_id"].str.fullmatch(r"LIDC-IDRI-\d{4}_\d+_n\d+").all()
    assert (df["image_path"] == "images/" + df["case_id"] + ".nii.gz").all()
    assert (df.filter(regex=r"^spiculation_r\d+$").notna().sum(axis=1) >= 3).all()
    # named terms decoded (not lost like the empty *_score columns)
    vals = set(df.filter(regex=r"^texture_r\d+$").stack().dropna().unique())
    assert vals <= {1, 2, 3, 4, 5} and (5 in vals or 1 in vals or 3 in vals)
    calc = set(df.filter(regex=r"^calcification_r\d+$").stack().dropna().unique())
    assert 6 in calc


def test_manifest_feeds_label_builder_and_loader(bucket, tmp_path):
    df = build_manifest(bucket, CFG)
    df.to_csv(tmp_path / "m.csv", index=False)
    loaded = load_manifest(tmp_path / "m.csv", bucket_root=bucket)
    assert all(p.startswith(str(bucket)) for p in loaded["image_path"])
    h = ConceptHierarchy.from_config(yaml.safe_load((files("hypertrace") / "configs/concepts/lidc.yaml").read_text()))
    lab = LabelBuilder(h).build(loaded)
    assert not np.isnan(lab.soft).all(axis=0).any()  # every concept has labels


def test_irregular_slice_gap_scans_dropped(bucket, tmp_path):
    for f in ("annotations.csv", "nodules.csv"):
        (tmp_path / f).write_text((bucket / f).read_text())
    scans = pd.read_csv(bucket / "scans.csv")
    bad_case = pd.read_csv(bucket / "nodules.csv").query("in_default_gt")["case_id"].iloc[0]
    scans.loc[scans.case_id == bad_case, "z_gap_max"] = 5.0
    scans.to_csv(tmp_path / "scans.csv", index=False)
    assert bad_case not in set(build_manifest(tmp_path, CFG)["case_id"])


def test_term_score_disagreement_is_caught(bucket, tmp_path):
    ann = pd.read_csv(bucket / "annotations.csv")
    i = ann.index[ann["spiculation_score"].notna()][0]
    ann.loc[i, "spiculation_score"] = 6 - ann.loc[i, "spiculation_score"] if ann.loc[i, "spiculation_score"] != 3 \
        else 1
    ann.to_csv(tmp_path / "annotations.csv", index=False)
    for f in ("nodules.csv", "scans.csv"):
        (tmp_path / f).write_text((bucket / f).read_text())
    with pytest.raises(MappingError, match="disagree"):
        build_manifest(tmp_path, CFG)


def test_cli_and_inspect(bucket, tmp_path, capsys):
    main(["--bucket-dir", str(bucket), "--out", str(tmp_path / "m.csv")])
    assert (tmp_path / "m.csv").exists()
    main(["--bucket-dir", str(bucket), "--inspect"])
    assert "## annotations.csv" in capsys.readouterr().out
