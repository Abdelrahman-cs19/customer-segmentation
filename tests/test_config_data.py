import hashlib

import pandas as pd
import pytest

from segmentation.config import load_config
from segmentation.data import audit_columns, audit_consistency, file_sha256, load_raw, top_correlations


def test_load_config_has_required_sections(cfg):
    assert cfg["project"]["seed"] == 42
    assert "engineered" in cfg["features"]["sets"]


def test_load_config_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "nope.yaml")


def test_load_config_missing_section_raises(tmp_path):
    p = tmp_path / "bad.yaml"
    p.write_text("project: {seed: 1}\n")
    with pytest.raises(ValueError):
        load_config(p)


def test_load_raw_roundtrip_and_validation(raw_df, cfg, tmp_path):
    path = tmp_path / "d.csv"
    raw_df.to_csv(path, index=False)
    out = load_raw(path, "CUST_ID", cfg["data"]["numeric_columns"])
    assert out.shape == raw_df.shape

    raw_df.drop(columns=["BALANCE"]).to_csv(path, index=False)
    with pytest.raises(ValueError, match="missing expected columns"):
        load_raw(path, "CUST_ID", cfg["data"]["numeric_columns"])

    dup = pd.concat([raw_df, raw_df.iloc[[0]]])
    dup.to_csv(path, index=False)
    with pytest.raises(ValueError, match="duplicated"):
        load_raw(path, "CUST_ID", cfg["data"]["numeric_columns"])


def test_load_raw_file_not_found(cfg, tmp_path):
    with pytest.raises(FileNotFoundError):
        load_raw(tmp_path / "x.csv", "CUST_ID", cfg["data"]["numeric_columns"])


def test_file_sha256_matches_hashlib(tmp_path):
    p = tmp_path / "a.bin"
    p.write_bytes(b"abc" * 1000)
    assert file_sha256(p) == hashlib.sha256(b"abc" * 1000).hexdigest()


def test_audit_consistency_counts_known_defects(raw_df):
    c = audit_consistency(raw_df)
    assert c["min_payments_missing"] == 3
    assert c["min_payments_missing_with_zero_payments"] == 2
    assert c["credit_limit_missing"] == 1
    assert c["cash_adv_frequency_above_1"] == 1
    assert c["purchases_component_mismatch_rows"] == 0


def test_audit_columns_and_correlations(raw_df):
    a = audit_columns(raw_df, "CUST_ID")
    assert set(a["column"]) == set(raw_df.columns) - {"CUST_ID"}
    assert a.set_index("column").loc["MINIMUM_PAYMENTS", "n_missing"] == 3
    pairs = top_correlations(raw_df.fillna(0), "CUST_ID", 0.5)
    assert {"feature_a", "feature_b", "abs_corr"} <= set(pairs.columns)
    assert (pairs["abs_corr"] >= 0.5).all()
