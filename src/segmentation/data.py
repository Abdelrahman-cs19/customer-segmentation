"""Raw data loading, schema validation and auditing."""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .logging_utils import get_logger

log = get_logger("data")


def file_sha256(path: str | Path) -> str:
    """Return the SHA-256 hex digest of a file (used to prove the raw data is untouched)."""
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_raw(path: str | Path, id_column: str, numeric_columns: list[str]) -> pd.DataFrame:
    """Load the raw CSV and validate its schema.

    The returned frame is a fresh copy; the file on disk is never modified.

    Raises:
        FileNotFoundError: If the file is missing.
        ValueError: If expected columns are missing, ids are duplicated/empty, or a numeric
            column cannot be parsed as numbers.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Raw data file not found: {path}")
    df = pd.read_csv(path)
    missing = [c for c in [id_column, *numeric_columns] if c not in df.columns]
    if missing:
        raise ValueError(f"Raw data is missing expected columns: {missing}")
    if df[id_column].isna().any():
        raise ValueError(f"Column {id_column} contains missing ids")
    if df[id_column].duplicated().any():
        raise ValueError(f"Column {id_column} contains duplicated ids")
    for col in numeric_columns:
        if not pd.api.types.is_numeric_dtype(df[col]):
            raise ValueError(f"Column {col} is not numeric (dtype={df[col].dtype})")
    log.info("Loaded %s rows x %s columns from %s", *df.shape, path.name)
    return df


def audit_columns(df: pd.DataFrame, id_column: str) -> pd.DataFrame:
    """Per-column audit table: dtype, missing, zeros, negatives, skew, spread and outlier counts."""
    num = df.drop(columns=[id_column])
    rows = []
    for c in num.columns:
        s = num[c]
        q1, q3 = s.quantile([0.25, 0.75])
        iqr = q3 - q1
        rows.append(
            {
                "column": c,
                "dtype": str(s.dtype),
                "n_missing": int(s.isna().sum()),
                "pct_missing": round(float(s.isna().mean() * 100), 3),
                "n_unique": int(s.nunique()),
                "pct_zero": round(float((s == 0).mean() * 100), 2),
                "n_negative": int((s < 0).sum()),
                "mean": float(s.mean()),
                "median": float(s.median()),
                "std": float(s.std()),
                "min": float(s.min()),
                "p99": float(s.quantile(0.99)),
                "max": float(s.max()),
                "skew": float(s.skew()),
                "n_outliers_3iqr": int((s > q3 + 3 * iqr).sum()),
                "top_value_share_pct": round(float(s.value_counts(normalize=True, dropna=True).iloc[0] * 100), 2),
            }
        )
    return pd.DataFrame(rows)


def audit_consistency(df: pd.DataFrame, tolerance: float = 0.01) -> dict[str, Any]:
    """Logical-consistency checks between related columns. Returns counts, not modified data."""
    purchase_gap = (df["PURCHASES"] - (df["ONEOFF_PURCHASES"] + df["INSTALLMENTS_PURCHASES"])).abs()
    miss = df["MINIMUM_PAYMENTS"].isna()
    return {
        "n_rows": int(len(df)),
        "n_duplicate_rows": int(df.duplicated().sum()),
        "purchases_equal_components_pct": round(float((purchase_gap <= tolerance).mean() * 100), 3),
        "purchases_component_mismatch_rows": int((purchase_gap > tolerance).sum()),
        "purchases_component_max_gap": float(purchase_gap.max()),
        "purchases_pos_but_zero_trx": int(((df["PURCHASES"] > 0) & (df["PURCHASES_TRX"] == 0)).sum()),
        "purchases_zero_but_pos_trx": int(((df["PURCHASES"] == 0) & (df["PURCHASES_TRX"] > 0)).sum()),
        "cash_adv_inconsistent_trx": int(
            (((df["CASH_ADVANCE"] > 0) & (df["CASH_ADVANCE_TRX"] == 0)) | ((df["CASH_ADVANCE"] == 0) & (df["CASH_ADVANCE_TRX"] > 0))).sum()
        ),
        "cash_adv_frequency_above_1": int((df["CASH_ADVANCE_FREQUENCY"] > 1).sum()),
        "balance_above_credit_limit": int((df["BALANCE"] > df["CREDIT_LIMIT"]).sum()),
        "min_payments_missing": int(miss.sum()),
        "min_payments_missing_with_zero_payments": int((miss & (df["PAYMENTS"] == 0)).sum()),
        "min_payments_missing_with_positive_payments": int((miss & (df["PAYMENTS"] > 0)).sum()),
        "zero_payments_total": int((df["PAYMENTS"] == 0).sum()),
        "credit_limit_missing": int(df["CREDIT_LIMIT"].isna().sum()),
        "fully_inactive_customers": int(((df["PURCHASES"] == 0) & (df["CASH_ADVANCE"] == 0)).sum()),
    }


def top_correlations(df: pd.DataFrame, id_column: str, threshold: float = 0.6, method: str = "spearman") -> pd.DataFrame:
    """Return feature pairs whose absolute correlation is at or above ``threshold``."""
    corr = df.drop(columns=[id_column]).corr(method=method).abs()
    mask = np.triu(np.ones(corr.shape, dtype=bool), k=1)
    pairs = corr.where(mask).stack().rename("abs_corr").reset_index()
    pairs.columns = ["feature_a", "feature_b", "abs_corr"]
    return pairs[pairs["abs_corr"] >= threshold].sort_values("abs_corr", ascending=False).reset_index(drop=True)
