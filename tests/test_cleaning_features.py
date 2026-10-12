import numpy as np
import pytest

from segmentation.cleaning import clean_data
from segmentation.features import ENGINEERED_FEATURES, add_engineered_features


def test_cleaning_never_drops_rows_and_leaves_no_nans(raw_df, cfg):
    clean, log = clean_data(raw_df, cfg)
    assert len(clean) == len(raw_df) == log["rows_after"]
    assert clean[cfg["data"]["numeric_columns"]].isna().sum().sum() == 0


def test_cleaning_does_not_mutate_input(raw_df, cfg):
    before = raw_df.copy()
    clean_data(raw_df, cfg)
    assert raw_df.equals(before)


def test_frequency_clipped_and_logged(raw_df, cfg):
    clean, log = clean_data(raw_df, cfg)
    assert clean["CASH_ADVANCE_FREQUENCY"].max() <= 1.0
    assert log["clipped_to_unit_interval"] == {"CASH_ADVANCE_FREQUENCY": 1}


def test_missing_flags_created_before_imputation(raw_df, cfg):
    clean, _ = clean_data(raw_df, cfg)
    assert clean["MIN_PAYMENTS_MISSING"].sum() == 3
    assert clean["CREDIT_LIMIT_MISSING"].sum() == 1


@pytest.mark.parametrize("strategy", ["median", "zero", "balance_ratio"])
def test_min_payment_strategies(raw_df, cfg, strategy):
    clean, log = clean_data(raw_df, cfg, min_payments_strategy=strategy)
    miss = raw_df["MINIMUM_PAYMENTS"].isna()
    assert clean.loc[~miss, "MINIMUM_PAYMENTS"].equals(raw_df.loc[~miss, "MINIMUM_PAYMENTS"])
    if strategy == "zero":
        assert (clean.loc[miss, "MINIMUM_PAYMENTS"] == 0).all()
    if strategy == "median":
        assert clean.loc[miss, "MINIMUM_PAYMENTS"].nunique() == 1
    assert log["min_payments_strategy"] == strategy


def test_unknown_strategy_raises(raw_df, cfg):
    with pytest.raises(ValueError):
        clean_data(raw_df, cfg, min_payments_strategy="magic")


def test_engineered_features_finite_and_correct(raw_df, cfg):
    clean, _ = clean_data(raw_df, cfg)
    f = add_engineered_features(clean)
    assert np.isfinite(f[ENGINEERED_FEATURES].to_numpy(dtype=float)).all()
    r = f.iloc[0]
    assert r["BALANCE_TO_LIMIT"] == pytest.approx(r["BALANCE"] / r["CREDIT_LIMIT"])
    comp = r["ONEOFF_PURCHASES"] + r["INSTALLMENTS_PURCHASES"]
    expected = r["ONEOFF_PURCHASES"] / comp if comp > 0 else 0.0
    assert r["ONEOFF_SHARE"] == pytest.approx(expected)
    assert f["ONEOFF_SHARE"].between(0, 1).all()


def test_engineered_zero_division_guards(raw_df, cfg):
    clean, _ = clean_data(raw_df, cfg)
    clean.loc[0, ["PURCHASES", "ONEOFF_PURCHASES", "INSTALLMENTS_PURCHASES", "PURCHASES_TRX", "CASH_ADVANCE", "CASH_ADVANCE_TRX"]] = 0
    clean.loc[0, "MINIMUM_PAYMENTS"] = 0.019
    f = add_engineered_features(clean)
    assert f.loc[0, "AVG_PURCHASE_TICKET"] == 0
    assert f.loc[0, "AVG_CASH_ADVANCE_TICKET"] == 0
    assert f.loc[0, "ONEOFF_SHARE"] == 0
    assert f.loc[0, "PAYMENT_TO_MINPAY_RATIO"] <= clean.loc[0, "PAYMENTS"]  # floor of 1 on the denominator
