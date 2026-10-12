"""Verifies the files produced by a real pipeline run (skipped if the pipeline has not been run)."""
import json
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
RAW = ROOT / "data" / "raw" / "Customer-Data - 2.csv"

pytestmark = pytest.mark.skipif(not (OUT / "run_summary.json").exists(), reason="run `python run_pipeline.py` first")


@pytest.fixture(scope="module")
def raw():
    return pd.read_csv(RAW)


@pytest.fixture(scope="module")
def assignments():
    return pd.read_csv(OUT / "customer_assignments.csv")


def test_expected_output_files_exist():
    for f in ["model_comparison.csv", "customer_assignments.csv", "segment_summary.csv", "preprocessing_comparison.csv",
              "clustering_grid_all_models.csv", "run_summary.json", "cleaning_log.json"]:
        assert (OUT / f).exists(), f
    assert (ROOT / "reports" / "final_report.md").exists()
    assert len(list((ROOT / "figures").glob("*.png"))) >= 19


def test_assignments_cover_every_customer_exactly_once(raw, assignments):
    assert len(assignments) == len(raw) == 8950
    assert assignments["CUST_ID"].is_unique
    assert set(assignments["CUST_ID"]) == set(raw["CUST_ID"])
    assert assignments["cluster"].notna().all() and (assignments["cluster"] >= 0).all()


def test_assignments_align_with_customer_attributes(raw, assignments):
    """Independent re-check: segment profile recomputed from RAW values via CUST_ID matches segment_summary.csv."""
    summ = pd.read_csv(OUT / "segment_summary.csv").set_index("cluster")
    m = raw.merge(assignments, on="CUST_ID", validate="one_to_one")
    assert m.groupby("cluster").size().to_dict() == summ["n_customers"].to_dict()
    for c, g in m.groupby("cluster"):
        assert g["PURCHASES"].median() == pytest.approx(summ.loc[c, "PURCHASES_median"], rel=1e-6)
        assert g["CASH_ADVANCE"].median() == pytest.approx(summ.loc[c, "CASH_ADVANCE_median"], rel=1e-6)


def test_segment_sizes_sum_and_names_unique():
    s = pd.read_csv(OUT / "segment_summary.csv")
    assert s["n_customers"].sum() == 8950
    assert s["segment_name"].is_unique


def test_model_comparison_has_selected_model_and_all_algorithms():
    t = pd.read_csv(OUT / "model_comparison.csv")
    assert t["selected"].sum() == 1
    assert {"kmeans", "agglomerative", "gmm", "dbscan", "hdbscan", "minibatch_kmeans"} <= set(t["algorithm"])
    sel = t[t["selected"]].iloc[0]
    assert sel["eligible"] and sel["stable"]


def test_run_summary_integrity_checks():
    r = json.loads((OUT / "run_summary.json").read_text())
    assert r["raw_data_unchanged"] is True
    assert r["alignment"]["n_label_mismatches"] == 0 and r["alignment"]["n_unassigned"] == 0
    assert r["alignment"]["same_id_set"] is True
    assert r.get("repredict_matches_labels", True) is True


def test_raw_file_matches_recorded_checksum():
    import hashlib

    recorded = (ROOT / "data" / "raw" / "SHA256SUM.txt").read_text().split()[0]
    assert hashlib.sha256(RAW.read_bytes()).hexdigest() == recorded


def test_scoring_reproduces_every_saved_assignment(raw, assignments):
    """End-to-end: scoring all 8,950 RAW rows with the saved bundle gives exactly the pipeline's assignments."""
    from segmentation.scoring import load_bundle, score_customers

    bundle = load_bundle(ROOT / "models" / "final_model.joblib")
    res, _ = score_customers(raw, bundle)
    merged = assignments.set_index("CUST_ID").loc[raw["CUST_ID"], "cluster"].to_numpy()
    assert (res["cluster"].to_numpy() == merged).all()
