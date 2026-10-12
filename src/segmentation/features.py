"""Credit-card behaviour feature engineering.

Ratios use guarded denominators so no infinities are produced. All features are computed from
the cleaned frame row by row (no cross-customer statistics), so they are leakage-free.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

ENGINEERED_FEATURES = [
    "BALANCE_TO_LIMIT",
    "PAYMENT_TO_MINPAY_RATIO",
    "ONEOFF_SHARE",
    "AVG_PURCHASE_TICKET",
    "AVG_CASH_ADVANCE_TICKET",
    "PAYMENT_TO_BALANCE",
]


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of ``df`` with engineered behaviour features appended.

    Features:
        BALANCE_TO_LIMIT: credit utilisation, BALANCE / CREDIT_LIMIT (limit >= 50 in the data).
        PAYMENT_TO_MINPAY_RATIO: PAYMENTS / max(MINIMUM_PAYMENTS, 1); the floor of 1 prevents the
            handful of near-zero minimum payments (min 0.019) from creating absurd ratios.
        ONEOFF_SHARE: share of purchase spend that is one-off, computed from the two components
            (not from PURCHASES) so it is robust to the 20 rows where the total disagrees.
        AVG_PURCHASE_TICKET: PURCHASES / PURCHASES_TRX, 0 when there are no transactions.
        AVG_CASH_ADVANCE_TICKET: CASH_ADVANCE / CASH_ADVANCE_TRX, 0 when there are none.
        PAYMENT_TO_BALANCE: PAYMENTS / (BALANCE + 1) (profiling only; not in the model feature sets).
    """
    out = df.copy()
    comp = out["ONEOFF_PURCHASES"] + out["INSTALLMENTS_PURCHASES"]
    out["BALANCE_TO_LIMIT"] = out["BALANCE"] / out["CREDIT_LIMIT"]
    out["PAYMENT_TO_MINPAY_RATIO"] = out["PAYMENTS"] / np.maximum(out["MINIMUM_PAYMENTS"], 1.0)
    out["ONEOFF_SHARE"] = np.where(comp > 0, out["ONEOFF_PURCHASES"] / comp.where(comp > 0, 1.0), 0.0)
    out["AVG_PURCHASE_TICKET"] = np.where(
        out["PURCHASES_TRX"] > 0, out["PURCHASES"] / out["PURCHASES_TRX"].where(out["PURCHASES_TRX"] > 0, 1), 0.0
    )
    out["AVG_CASH_ADVANCE_TICKET"] = np.where(
        out["CASH_ADVANCE_TRX"] > 0, out["CASH_ADVANCE"] / out["CASH_ADVANCE_TRX"].where(out["CASH_ADVANCE_TRX"] > 0, 1), 0.0
    )
    out["PAYMENT_TO_BALANCE"] = out["PAYMENTS"] / (out["BALANCE"] + 1.0)
    new_cols = out[ENGINEERED_FEATURES]
    if not np.isfinite(new_cols.to_numpy(dtype=float)).all():
        raise ValueError("Engineered features contain NaN or infinite values")
    return out
