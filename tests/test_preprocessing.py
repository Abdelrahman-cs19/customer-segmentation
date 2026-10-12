import numpy as np
import pandas as pd
import pytest

from segmentation.cleaning import clean_data
from segmentation.experiments import make_matrix
from segmentation.features import add_engineered_features
from segmentation.preprocessing import build_preprocessor, high_correlation_pairs, select_log_columns, validate_transformed


@pytest.fixture()
def feats(raw_df, cfg):
    return add_engineered_features(clean_data(raw_df, cfg)[0])


def test_select_log_columns_skips_proportions_and_symmetric(feats):
    cols = select_log_columns(feats[["PURCHASES_FREQUENCY", "BALANCE", "CASH_ADVANCE", "PRC_FULL_PAYMENT"]], 1.0)
    assert "PURCHASES_FREQUENCY" not in cols and "PRC_FULL_PAYMENT" not in cols
    assert "CASH_ADVANCE" in cols


@pytest.mark.parametrize("scaler", ["standard", "robust"])
@pytest.mark.parametrize("log_on", [True, False])
def test_pipeline_output_is_finite_and_validated(feats, cfg, scaler, log_on):
    cols = cfg["features"]["sets"]["engineered"]
    Xt, _ = make_matrix(feats, cols, cfg, log_on, scaler)
    assert Xt.shape == (len(feats), len(cols))
    assert np.isfinite(Xt.to_numpy()).all()


def test_standard_scaler_gives_zero_mean_unit_variance(feats, cfg):
    Xt, _ = make_matrix(feats, cfg["features"]["sets"]["base_no_total"], cfg, True, "standard")
    assert np.allclose(Xt.mean(), 0, atol=1e-8)
    assert np.allclose(Xt.std(ddof=0), 1, atol=1e-6)


def test_constant_feature_removed_by_variance_filter(feats):
    d = feats[["BALANCE", "PAYMENTS"]].copy()
    d["CONST"] = 5.0
    pre = build_preprocessor(["BALANCE", "PAYMENTS"], "standard")
    out = pre.fit_transform(d)
    assert "CONST" not in out.columns and out.shape[1] == 2


def test_pipeline_is_reproducible(feats, cfg):
    cols = cfg["features"]["sets"]["engineered"]
    a, _ = make_matrix(feats, cols, cfg, True, "robust")
    b, _ = make_matrix(feats, cols, cfg, True, "robust")
    pd.testing.assert_frame_equal(a, b)


def test_unknown_scaler_raises():
    with pytest.raises(ValueError):
        build_preprocessor([], "minmax")


def test_validate_transformed_rejects_bad_matrices():
    good = pd.DataFrame({"a": [0.0, 1.0, 2.0], "b": [1.0, 0.0, 3.0]})
    validate_transformed(good, 3)
    with pytest.raises(ValueError, match="NaN"):
        validate_transformed(good.assign(a=[0.0, np.nan, 1.0]), 3)
    with pytest.raises(ValueError, match="Row count"):
        validate_transformed(good, 5)
    with pytest.raises(ValueError, match="Constant"):
        validate_transformed(good.assign(a=1.0), 3)


def test_high_correlation_pairs_detects_duplicate_feature():
    rng = np.random.default_rng(1)
    x = rng.normal(size=200)
    d = pd.DataFrame({"a": x, "b": x * 2 + 0.001 * rng.normal(size=200), "c": rng.normal(size=200)})
    pairs = high_correlation_pairs(d, 0.9)
    assert len(pairs) == 1 and set(pairs.iloc[0][["feature_a", "feature_b"]]) == {"a", "b"}
