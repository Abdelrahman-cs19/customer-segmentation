import numpy as np
import pandas as pd
import pytest

from segmentation.profiling import (
    build_assignments, name_segments, profile_index, segment_summary, verify_alignment,
)
from segmentation.selection import pick_finalists, rank_and_select, screen_eligibility


def _grid():
    return pd.DataFrame(
        {
            "model_id": ["a", "b", "c", "d"],
            "algorithm": ["kmeans", "kmeans", "agglomerative", "hdbscan"],
            "params": [{}, {}, {}, {}],
            "n_clusters": [4, 12, 5, 4],
            "noise_pct": [0.0, 0.0, 0.0, 45.0],
            "min_cluster_pct": [13.0, 5.0, 0.01, 2.0],
            "max_cluster_pct": [50.0, 20.0, 99.0, 40.0],
            "silhouette": [0.30, 0.25, 0.40, 0.50],
            "davies_bouldin": [1.2, 1.4, 0.5, 0.9],
            "calinski_harabasz": [3000.0, 2000.0, 100.0, 800.0],
        }
    )


def test_screen_eligibility_flags_each_failure_mode(cfg):
    s = screen_eligibility(_grid(), cfg).set_index("model_id")
    assert s.loc["a", "eligible"]
    assert "outside" in s.loc["b", "reject_reason"]
    assert "smallest cluster" in s.loc["c", "reject_reason"]
    assert "noise" in s.loc["d", "reject_reason"]


def test_pick_finalists_includes_best_ineligible_for_transparency(cfg):
    s = screen_eligibility(_grid(), cfg)
    f = pick_finalists(s, per_algorithm=2)
    # kmeans -> its one eligible model (a); agglomerative and hdbscan have none eligible -> best ineligible kept
    assert set(f["model_id"]) == {"a", "c", "d"}


def test_rank_and_select_only_ranks_eligible_stable(cfg):
    s = screen_eligibility(_grid(), cfg)
    s["stability_ari_mean"] = [0.99, 0.99, 0.99, 0.2]
    s["stability_ari_std"] = 0.01
    t = rank_and_select(s, cfg)
    pool = t[t["in_pool"]]
    assert list(pool["model_id"]) == ["a"]
    assert t.loc[t["model_id"] == "d", "mean_rank"].isna().all()


@pytest.fixture()
def prof_df():
    rng = np.random.default_rng(3)
    n = 120
    d = pd.DataFrame({"CUST_ID": [f"C{i}" for i in range(n)]})
    from segmentation.profiling import PROFILE_FEATURES

    for f in PROFILE_FEATURES:
        d[f] = rng.random(n) + 0.1
    d["PURCHASES"] = np.where(np.arange(n) % 2 == 0, 0.0, 5.0)
    d["PAYMENT_TO_MINPAY_RATIO"] = rng.random(n) * 3
    return d


def test_build_assignments_alignment_and_mismatch_detection(prof_df):
    labels = np.arange(len(prof_df)) % 3
    names = {0: "A", 1: "B", 2: "C"}
    a = build_assignments(prof_df, labels, "CUST_ID", names)
    v = verify_alignment(a, prof_df, labels, "CUST_ID")
    assert v["n_label_mismatches"] == 0 and v["same_id_set"] and v["row_order_preserved"]
    # shuffle the assignment table rows: lookup by id must still match
    shuffled = a.sample(frac=1, random_state=0)
    assert verify_alignment(shuffled, prof_df, labels, "CUST_ID")["n_label_mismatches"] == 0
    # corrupt a label: must be caught
    bad = a.copy()
    bad.loc[0, "cluster"] = 2
    assert verify_alignment(bad, prof_df, labels, "CUST_ID")["n_label_mismatches"] == 1


def test_build_assignments_rejects_length_mismatch(prof_df):
    with pytest.raises(ValueError):
        build_assignments(prof_df, np.zeros(5, dtype=int), "CUST_ID", {0: "A"})


def test_segment_summary_sizes_sum_to_total(prof_df):
    labels = np.arange(len(prof_df)) % 3
    s = segment_summary(prof_df, labels, {0: "A", 1: "B", 2: "C"})
    assert s["n_customers"].sum() == len(prof_df)
    assert s["pct_customers"].sum() == pytest.approx(100, abs=0.1)


def test_profile_index_of_whole_population_is_100(prof_df):
    idx = profile_index(prof_df, np.zeros(len(prof_df), dtype=int))
    assert (idx.iloc[0].dropna() == 100).all()


def test_name_segments_rules(cfg):
    base = {"cluster": 0, "PURCHASES_FREQUENCY_median": 0.0, "CASH_ADVANCE_FREQUENCY_median": 0.25, "PRC_FULL_PAYMENT_median": 0.0, "BALANCE_TO_LIMIT_median": 0.7}
    rows = [
        base,
        {**base, "cluster": 1, "PURCHASES_FREQUENCY_median": 0.9, "CASH_ADVANCE_FREQUENCY_median": 0.0, "PRC_FULL_PAYMENT_median": 0.8},
        {**base, "cluster": 2, "PURCHASES_FREQUENCY_median": 0.1, "CASH_ADVANCE_FREQUENCY_median": 0.0, "BALANCE_TO_LIMIT_median": 0.01},
        {**base, "cluster": 3, "PURCHASES_FREQUENCY_median": 0.7, "CASH_ADVANCE_FREQUENCY_median": 0.0},
    ]
    names = name_segments(pd.DataFrame(rows), cfg)
    assert names == {0: "Cash-Advance Borrowers", 1: "Frequent Full-Payers", 2: "Low-Activity Light Users", 3: "Active Balance-Carriers"}


def test_name_segments_unique_when_two_match(cfg):
    r = {"PURCHASES_FREQUENCY_median": 0.0, "CASH_ADVANCE_FREQUENCY_median": 0.3, "PRC_FULL_PAYMENT_median": 0.0, "BALANCE_TO_LIMIT_median": 0.7}
    names = name_segments(pd.DataFrame([{**r, "cluster": 0}, {**r, "cluster": 1}]), cfg)
    assert len(set(names.values())) == 2
