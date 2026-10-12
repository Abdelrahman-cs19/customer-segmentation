"""Reproducible scikit-learn preprocessing: impute -> (log1p) -> scale -> variance filter."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, RobustScaler, StandardScaler

from .logging_utils import get_logger

log = get_logger("preprocessing")


def select_log_columns(X: pd.DataFrame, skew_threshold: float = 1.0) -> list[str]:
    """Pick columns eligible for ``log1p``.

    A column qualifies if it is non-negative, unbounded (max > 1, i.e. not a proportion) and
    right-skewed above ``skew_threshold``. Proportions in [0, 1] are left alone because log1p
    barely changes them and would only blur their interpretation.
    """
    eligible = []
    for c in X.columns:
        s = X[c].dropna()
        if s.min() >= 0 and s.max() > 1.0 and s.skew() > skew_threshold:
            eligible.append(c)
    return eligible


def build_preprocessor(
    log_columns: list[str],
    scaler: str = "standard",
    variance_threshold: float = 1e-8,
) -> Pipeline:
    """Build the preprocessing pipeline (pandas-in, pandas-out).

    Args:
        log_columns: Columns to transform with ``log1p`` (may be empty).
        scaler: ``"standard"`` or ``"robust"``.
        variance_threshold: Features with variance <= this after scaling are dropped (constant features).
    """
    if scaler == "standard":
        scale_step = StandardScaler()
    elif scaler == "robust":
        scale_step = RobustScaler(quantile_range=(25.0, 75.0))
    else:
        raise ValueError(f"Unknown scaler '{scaler}' (expected 'standard' or 'robust')")

    log_tf = FunctionTransformer(np.log1p, feature_names_out="one-to-one", validate=False)
    column_step = ColumnTransformer(
        [("log1p", log_tf, list(log_columns))],
        remainder="passthrough",
        verbose_feature_names_out=False,
    )
    pipe = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),  # safety net; cleaning already imputed
            ("log", column_step),
            ("scale", scale_step),
            ("variance", VarianceThreshold(threshold=variance_threshold)),
        ]
    )
    pipe.set_output(transform="pandas")
    return pipe


def validate_transformed(Xt: pd.DataFrame, n_rows: int, max_abs: float = 1e6) -> None:
    """Fail loudly if the transformed matrix is not safe to cluster.

    Checks: row count preserved, no NaN/inf, no constant columns, magnitudes sane.
    """
    if len(Xt) != n_rows:
        raise ValueError(f"Row count changed during preprocessing: {n_rows} -> {len(Xt)}")
    arr = Xt.to_numpy(dtype=float)
    if not np.isfinite(arr).all():
        raise ValueError("Transformed data contains NaN or infinite values")
    if (Xt.std(ddof=0) < 1e-12).any():
        raise ValueError(f"Constant columns after preprocessing: {list(Xt.columns[Xt.std(ddof=0) < 1e-12])}")
    if np.abs(arr).max() > max_abs:
        raise ValueError(f"Transformed values exceed {max_abs:g}")


def high_correlation_pairs(Xt: pd.DataFrame, threshold: float = 0.9) -> pd.DataFrame:
    """List feature pairs with |Pearson r| >= threshold in the transformed space."""
    corr = Xt.corr().abs()
    mask = np.triu(np.ones(corr.shape, dtype=bool), k=1)
    pairs = corr.where(mask).stack().reset_index()
    pairs.columns = ["feature_a", "feature_b", "abs_corr"]
    return pairs[pairs["abs_corr"] >= threshold].sort_values("abs_corr", ascending=False).reset_index(drop=True)
