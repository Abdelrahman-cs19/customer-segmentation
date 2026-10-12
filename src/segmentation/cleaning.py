"""Deterministic, documented data-cleaning steps.

Rows are never dropped: high-value customers are legitimate observations. Every change is
logged in the returned ``cleaning_log`` so the report can state exactly what was touched.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .logging_utils import get_logger

log = get_logger("cleaning")

VALID_MIN_PAYMENT_STRATEGIES = ("median", "zero", "balance_ratio")


def clean_data(df: pd.DataFrame, cfg: dict[str, Any], min_payments_strategy: str | None = None) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Clean the raw frame without mutating it.

    Steps:
        1. Clip proportion columns to [0, 1] (8 rows have CASH_ADVANCE_FREQUENCY > 1).
        2. Add ``MIN_PAYMENTS_MISSING`` / ``CREDIT_LIMIT_MISSING`` indicator flags *before* imputing.
        3. Impute CREDIT_LIMIT (median) and MINIMUM_PAYMENTS (configurable strategy).
        4. Add ``PURCHASE_COMPONENT_MISMATCH`` flag (informational, not a modelling feature).

    Args:
        df: Raw dataframe.
        cfg: Project config.
        min_payments_strategy: Override for the MINIMUM_PAYMENTS strategy (used for sensitivity runs).

    Returns:
        (clean dataframe, cleaning log dict)
    """
    ccfg = cfg["cleaning"]
    strategy = min_payments_strategy or ccfg["min_payments_strategy"]
    if strategy not in VALID_MIN_PAYMENT_STRATEGIES:
        raise ValueError(f"Unknown min_payments_strategy '{strategy}'. Choose from {VALID_MIN_PAYMENT_STRATEGIES}")

    out = df.copy()
    cleaning_log: dict[str, Any] = {"min_payments_strategy": strategy, "rows_before": int(len(out))}

    # 1. Proportions must lie in [0, 1]
    clipped: dict[str, int] = {}
    for col in ccfg["proportion_columns"]:
        n_bad = int(((out[col] < 0) | (out[col] > 1)).sum())
        if n_bad:
            out[col] = out[col].clip(0, 1)
            clipped[col] = n_bad
    cleaning_log["clipped_to_unit_interval"] = clipped

    # 2. Missingness indicators (before imputation so the information is preserved)
    out["MIN_PAYMENTS_MISSING"] = out["MINIMUM_PAYMENTS"].isna().astype(int)
    out["CREDIT_LIMIT_MISSING"] = out["CREDIT_LIMIT"].isna().astype(int)

    # 3a. CREDIT_LIMIT: a single missing value -> median is harmless and transparent
    n_cl = int(out["CREDIT_LIMIT"].isna().sum())
    if n_cl:
        cl_median = float(out["CREDIT_LIMIT"].median())
        out["CREDIT_LIMIT"] = out["CREDIT_LIMIT"].fillna(cl_median)
        cleaning_log["credit_limit_imputed"] = {"n": n_cl, "value": cl_median}

    # 3b. MINIMUM_PAYMENTS
    n_mp = int(out["MINIMUM_PAYMENTS"].isna().sum())
    if n_mp:
        miss = out["MINIMUM_PAYMENTS"].isna()
        if strategy == "median":
            fill_value: Any = float(out["MINIMUM_PAYMENTS"].median())
            out.loc[miss, "MINIMUM_PAYMENTS"] = fill_value
        elif strategy == "zero":
            fill_value = 0.0
            out.loc[miss, "MINIMUM_PAYMENTS"] = 0.0
        else:  # balance_ratio: scale the median min-payment-to-balance ratio by the customer's balance
            valid = out.loc[~miss & (out["BALANCE"] > 0)]
            ratio = float((valid["MINIMUM_PAYMENTS"] / valid["BALANCE"]).median())
            fill_value = ratio
            out.loc[miss, "MINIMUM_PAYMENTS"] = out.loc[miss, "BALANCE"] * ratio
        cleaning_log["min_payments_imputed"] = {"n": n_mp, "strategy": strategy, "fill": fill_value}

    # 4. Informational consistency flag
    gap = (out["PURCHASES"] - (out["ONEOFF_PURCHASES"] + out["INSTALLMENTS_PURCHASES"])).abs()
    out["PURCHASE_COMPONENT_MISMATCH"] = (gap > ccfg["purchase_sum_tolerance"]).astype(int)

    cleaning_log["rows_after"] = int(len(out))
    if cleaning_log["rows_before"] != cleaning_log["rows_after"]:
        raise RuntimeError("Cleaning must never drop rows")
    remaining = int(out[cfg["data"]["numeric_columns"]].isna().sum().sum())
    if remaining:
        raise RuntimeError(f"{remaining} missing values remain after cleaning")
    log.info("Cleaning done: %s", cleaning_log)
    return out, cleaning_log
