import numpy as np
import pandas as pd
import pytest
from sklearn.cluster import KMeans

from segmentation.cleaning import clean_data
from segmentation.experiments import make_matrix
from segmentation.features import add_engineered_features
from segmentation.scoring import prepare_input, score_customers

INPUT_COLS = ["BALANCE", "BALANCE_FREQUENCY", "ONEOFF_PURCHASES", "INSTALLMENTS_PURCHASES", "CASH_ADVANCE", "PURCHASES_FREQUENCY",
              "ONEOFF_PURCHASES_FREQUENCY", "PURCHASES_INSTALLMENTS_FREQUENCY", "CASH_ADVANCE_FREQUENCY", "CASH_ADVANCE_TRX",
              "PURCHASES_TRX", "CREDIT_LIMIT", "PAYMENTS", "MINIMUM_PAYMENTS", "PRC_FULL_PAYMENT"]


@pytest.fixture()
def trained(raw_df, cfg):
    """Mimic what the pipeline stores: preprocessor, KMeans and training-time fill values."""
    clean, log = clean_data(raw_df, cfg)
    df = add_engineered_features(clean)
    cols = cfg["features"]["sets"]["engineered"]
    Xt, pre = make_matrix(df, cols, cfg, True, "robust")
    km = KMeans(n_clusters=3, n_init=5, random_state=0).fit(Xt.to_numpy())
    bundle = {
        "preprocessor": pre, "model": km, "feature_columns": cols,
        "segment_names": {0: "A", 1: "B", 2: "C"},
        "fill_values": {"CREDIT_LIMIT": float(raw_df["CREDIT_LIMIT"].median()),
                        "MINIMUM_PAYMENTS": {"strategy": "median", "value": log["min_payments_imputed"]["fill"]}},
        "input_columns": INPUT_COLS,
        "input_ranges": {c: {"max": float(df[c].max())} for c in INPUT_COLS},
        "proportion_columns": cfg["cleaning"]["proportion_columns"],
    }
    return bundle, km.labels_


def test_scoring_reproduces_training_labels_from_raw_rows(trained, raw_df):
    """Train/serve consistency: raw rows (with NaNs and a frequency > 1) must land in their training clusters."""
    bundle, train_labels = trained
    res, _ = score_customers(raw_df, bundle)
    assert (res["cluster"].to_numpy() == train_labels).all()


def test_single_row_equals_same_row_in_batch(trained, raw_df):
    bundle, _ = trained
    batch, _ = score_customers(raw_df, bundle)
    for i in (0, 3, 9, 50):
        single, _ = score_customers(raw_df.iloc[[i]], bundle)
        assert single["cluster"].iloc[0] == batch["cluster"].iloc[i]


def test_missing_values_use_training_constants_not_the_input_frame(trained, raw_df):
    bundle, _ = trained
    row = raw_df.iloc[[0]].copy()
    row["MINIMUM_PAYMENTS"] = np.nan
    row["CREDIT_LIMIT"] = np.nan
    prepared, notes = prepare_input(row, bundle)
    assert prepared["MINIMUM_PAYMENTS"].iloc[0] == pytest.approx(bundle["fill_values"]["MINIMUM_PAYMENTS"]["value"])
    assert prepared["CREDIT_LIMIT"].iloc[0] == pytest.approx(bundle["fill_values"]["CREDIT_LIMIT"])
    assert any("MINIMUM_PAYMENTS" in n for n in notes) and any("CREDIT_LIMIT" in n for n in notes)


def test_input_is_not_mutated(trained, raw_df):
    bundle, _ = trained
    before = raw_df.copy()
    score_customers(raw_df, bundle)
    assert raw_df.equals(before)


def test_margin_and_distances(trained, raw_df):
    bundle, _ = trained
    res, _ = score_customers(raw_df, bundle)
    assert res["margin"].between(0, 1).all()
    d = res[[c for c in res.columns if c.startswith("dist_")]]
    assert d.shape[1] == 3 and (d.idxmin(axis=1).str.replace("dist_", "") == res["segment_name"]).all()


def test_proportion_above_one_is_clipped_with_note(trained, raw_df):
    bundle, _ = trained
    row = raw_df.iloc[[1]].copy()
    row["PURCHASES_FREQUENCY"] = 3.0
    _, notes = prepare_input(row, bundle)
    assert any("PURCHASES_FREQUENCY" in n for n in notes)


@pytest.mark.parametrize("mutate,msg", [
    (lambda d: d.drop(columns=["BALANCE"]), "Missing required"),
    (lambda d: d.assign(BALANCE=-5.0), "Negative"),
    (lambda d: d.assign(PAYMENTS="abc"), "numeric"),
    (lambda d: d.assign(PAYMENTS=np.nan), "numeric"),
    (lambda d: d.assign(CREDIT_LIMIT=0.0), "positive"),
])
def test_invalid_inputs_raise(trained, raw_df, mutate, msg):
    bundle, _ = trained
    with pytest.raises(ValueError, match=msg):
        score_customers(mutate(raw_df.iloc[:3].copy()), bundle)
