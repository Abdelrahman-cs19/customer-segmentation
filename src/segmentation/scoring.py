"""Score new customers with the saved model bundle.

The training pipeline computes imputation values from the full training frame. For scoring, those
values are stored in the bundle (``fill_values``) and re-used, so a single customer is treated exactly
as a training customer with the same inputs would have been.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from .features import add_engineered_features


def load_bundle(path: str | Path) -> dict[str, Any]:
    """Load the model bundle written by the pipeline.

    Raises:
        FileNotFoundError: if the file is missing.
        KeyError: if the bundle predates scoring support (re-run the pipeline).
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Model bundle not found: {path}. Run `python run_pipeline.py` first.")
    bundle = joblib.load(path)
    needed = {"preprocessor", "model", "feature_columns", "segment_names", "fill_values", "input_columns"}
    missing = needed - bundle.keys()
    if missing:
        raise KeyError(f"Model bundle is missing {sorted(missing)}; re-run the pipeline to rebuild it.")
    return bundle


def prepare_input(df: pd.DataFrame, bundle: dict[str, Any]) -> tuple[pd.DataFrame, list[str]]:
    """Validate and clean raw customer rows the same way training data was cleaned.

    Returns the cleaned frame (with engineered features) and a list of human-readable notes about
    anything that was changed or looks unusual. Never mutates ``df``.

    Raises:
        ValueError: for missing columns, non-numeric values, or negative amounts.
    """
    notes: list[str] = []
    required = [c for c in bundle["input_columns"] if c not in ("CREDIT_LIMIT", "MINIMUM_PAYMENTS")]
    absent = [c for c in required if c not in df.columns]
    if absent:
        raise ValueError(f"Missing required columns: {absent}")
    work = df.copy()
    for c in bundle["input_columns"]:
        if c not in work.columns:
            work[c] = np.nan
        work[c] = pd.to_numeric(work[c], errors="coerce") if c in ("CREDIT_LIMIT", "MINIMUM_PAYMENTS") else work[c]
    num = work[bundle["input_columns"]]
    non_numeric = [c for c in required if not pd.api.types.is_numeric_dtype(work[c]) or work[c].isna().any()]
    if non_numeric:
        raise ValueError(f"Columns must be numeric with no blanks (only CREDIT_LIMIT and MINIMUM_PAYMENTS may be empty): {non_numeric}")
    neg = [c for c in bundle["input_columns"] if (num[c] < 0).any()]
    if neg:
        raise ValueError(f"Negative values are not valid for: {neg}")

    for c in bundle.get("proportion_columns", []):
        if c in work.columns:
            n_bad = int((work[c] > 1).sum())
            if n_bad:
                work[c] = work[c].clip(0, 1)
                notes.append(f"{c}: {n_bad} value(s) above 1 clipped to 1 (as in training)")

    n_cl = int(work["CREDIT_LIMIT"].isna().sum())
    if n_cl:
        work["CREDIT_LIMIT"] = work["CREDIT_LIMIT"].fillna(bundle["fill_values"]["CREDIT_LIMIT"])
        notes.append(f"CREDIT_LIMIT missing for {n_cl} row(s): filled with the training median ({bundle['fill_values']['CREDIT_LIMIT']:,.0f})")
    if (work["CREDIT_LIMIT"] <= 0).any():
        raise ValueError("CREDIT_LIMIT must be positive")

    n_mp = int(work["MINIMUM_PAYMENTS"].isna().sum())
    if n_mp:
        fv = bundle["fill_values"]["MINIMUM_PAYMENTS"]
        miss = work["MINIMUM_PAYMENTS"].isna()
        work.loc[miss, "MINIMUM_PAYMENTS"] = work.loc[miss, "BALANCE"] * fv["value"] if fv["strategy"] == "balance_ratio" else fv["value"]
        notes.append(f"MINIMUM_PAYMENTS missing for {n_mp} row(s): filled using the training '{fv['strategy']}' rule")

    if "PURCHASES" not in work.columns:
        work["PURCHASES"] = work["ONEOFF_PURCHASES"] + work["INSTALLMENTS_PURCHASES"]

    for c, r in bundle.get("input_ranges", {}).items():
        if c in work.columns and (work[c] > r["max"]).any():
            notes.append(f"{c}: value above the largest seen in training ({r['max']:,.0f}); the result is an extrapolation")
    if ((work["PURCHASES"] > 0) & (work["PURCHASES_FREQUENCY"] == 0)).any():
        notes.append("Purchases > 0 but purchase frequency = 0: inconsistent inputs")
    if ((work["CASH_ADVANCE"] > 0) & (work["CASH_ADVANCE_TRX"] == 0)).any():
        notes.append("Cash advance > 0 but 0 cash-advance transactions: inconsistent inputs")
    return add_engineered_features(work), notes


def score_customers(df: pd.DataFrame, bundle: dict[str, Any]) -> tuple[pd.DataFrame, list[str]]:
    """Assign each row to a segment.

    Returns:
        (results, notes). ``results`` has ``cluster``, ``segment_name``, one ``dist_<segment>`` column per
        segment (Euclidean distance to the segment centroid in the model's input space) and ``margin``
        = 1 - nearest/second-nearest distance (0 = tie between two segments, 1 = clearly one segment).
        ``margin`` is a heuristic closeness indicator, not a probability.
    """
    prepared, notes = prepare_input(df, bundle)
    Xt = bundle["preprocessor"].transform(prepared[bundle["feature_columns"]])
    X = Xt.to_numpy(dtype=float)
    if not np.isfinite(X).all():
        raise ValueError("Transformed input contains NaN or infinite values")
    model = bundle["model"]
    labels = np.asarray(model.predict(X)).astype(int)
    names = bundle["segment_names"]
    out = pd.DataFrame({"cluster": labels, "segment_name": [names[int(c)] for c in labels]}, index=df.index)
    if hasattr(model, "cluster_centers_"):
        centers = model.cluster_centers_
        d = np.linalg.norm(X[:, None, :] - centers[None, :, :], axis=2)
        for c in range(d.shape[1]):
            out[f"dist_{names[c]}"] = d[:, c]
        ds = np.sort(d, axis=1)
        out["margin"] = 1 - ds[:, 0] / np.where(ds[:, 1] > 0, ds[:, 1], np.nan)
    return out, notes
