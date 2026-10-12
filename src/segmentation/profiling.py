"""Customer assignments, segment profiling, rule-based naming and recommendations.

Everything here describes *observed* behaviour in the data. The dataset has no revenue, interest,
fee, or delinquency fields, so statements about profitability or credit risk are labelled as
assumptions and kept separate from the measured profile.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

PROFILE_FEATURES = [
    "BALANCE", "BALANCE_FREQUENCY", "PURCHASES", "ONEOFF_PURCHASES", "INSTALLMENTS_PURCHASES", "CASH_ADVANCE",
    "PURCHASES_FREQUENCY", "ONEOFF_PURCHASES_FREQUENCY", "PURCHASES_INSTALLMENTS_FREQUENCY", "CASH_ADVANCE_FREQUENCY",
    "CASH_ADVANCE_TRX", "PURCHASES_TRX", "CREDIT_LIMIT", "PAYMENTS", "MINIMUM_PAYMENTS", "PRC_FULL_PAYMENT", "TENURE",
    "BALANCE_TO_LIMIT", "PAYMENT_TO_MINPAY_RATIO", "ONEOFF_SHARE", "AVG_PURCHASE_TICKET", "AVG_CASH_ADVANCE_TICKET",
]


def build_assignments(df: pd.DataFrame, labels: np.ndarray, id_column: str, names: dict[int, str], pca_xy: np.ndarray | None = None) -> pd.DataFrame:
    """Assemble the customer -> segment table. ``labels`` must be in the same row order as ``df``."""
    if len(df) != len(labels):
        raise ValueError(f"Row/label mismatch: {len(df)} customers vs {len(labels)} labels")
    out = pd.DataFrame({id_column: df[id_column].to_numpy(), "cluster": np.asarray(labels).astype(int)})
    out["segment_name"] = out["cluster"].map(names)
    if pca_xy is not None:
        out["pca_1"], out["pca_2"] = pca_xy[:, 0], pca_xy[:, 1]
    if out[id_column].duplicated().any():
        raise ValueError("Duplicate customer ids in assignments")
    return out


def verify_alignment(assignments: pd.DataFrame, source: pd.DataFrame, labels: np.ndarray, id_column: str) -> dict[str, Any]:
    """Prove customer ids and cluster labels stayed aligned (checks *every* row, not a sample).

    Args:
        assignments: the final customer -> cluster table.
        source: the frame the labels were produced from, in the order they were produced.
        labels: raw label array in ``source`` row order.

    Checks: identical id set; all ids assigned; each id's recorded cluster equals the label at that
    id's original row position; row order preserved.
    """
    labels = np.asarray(labels).astype(int)
    lookup = assignments.set_index(id_column)["cluster"]
    recorded = lookup.reindex(source[id_column]).to_numpy()
    return {
        "same_id_set": bool(set(assignments[id_column]) == set(source[id_column])),
        "n_customers": int(len(source)),
        "n_unassigned": int(pd.isna(recorded).sum()),
        "n_label_mismatches": int((recorded != labels).sum()),
        "row_order_preserved": bool((assignments[id_column].to_numpy() == source[id_column].to_numpy()).all()),
    }


def fit_pca(X: np.ndarray, seed: int, n_components: int = 2) -> tuple[PCA, np.ndarray]:
    """Fit PCA on the transformed matrix; returns (pca, projected coordinates)."""
    pca = PCA(n_components=n_components, random_state=seed)
    return pca, pca.fit_transform(X)


def segment_summary(df: pd.DataFrame, labels: np.ndarray, names: dict[int, str]) -> pd.DataFrame:
    """One row per segment with size, mean/median of key features and observed repayment indicators."""
    d = df.copy()
    d["cluster"] = np.asarray(labels).astype(int)
    rows = []
    for c, g in d.groupby("cluster"):
        row: dict[str, Any] = {
            "cluster": int(c),
            "segment_name": names[int(c)],
            "n_customers": int(len(g)),
            "pct_customers": round(len(g) / len(d) * 100, 2),
        }
        for f in PROFILE_FEATURES:
            row[f"{f}_mean"] = float(g[f].mean())
            row[f"{f}_median"] = float(g[f].median())
        # Observed repayment / usage shares
        row["pct_use_cash_advance"] = float((g["CASH_ADVANCE"] > 0).mean() * 100)
        row["pct_no_purchases"] = float((g["PURCHASES"] == 0).mean() * 100)
        row["pct_full_payers_ge_50"] = float((g["PRC_FULL_PAYMENT"] >= 0.5).mean() * 100)
        row["pct_never_full_payment"] = float((g["PRC_FULL_PAYMENT"] == 0).mean() * 100)
        row["pct_payments_below_min_due"] = float((g["PAYMENT_TO_MINPAY_RATIO"] < 1).mean() * 100)
        row["pct_balance_above_limit"] = float((g["BALANCE"] > g["CREDIT_LIMIT"]).mean() * 100)
        rows.append(row)
    return pd.DataFrame(rows)


def profile_index(df: pd.DataFrame, labels: np.ndarray, features: list[str] | None = None) -> pd.DataFrame:
    """Segment mean / population mean x 100 for each feature (100 = average customer)."""
    feats = features or PROFILE_FEATURES
    d = df[feats].copy()
    d["cluster"] = np.asarray(labels).astype(int)
    seg = d.groupby("cluster")[feats].mean()
    overall = d[feats].mean().replace(0, np.nan)
    return (seg / overall * 100).round(1)


def name_segments(summary: pd.DataFrame, cfg: dict[str, Any]) -> dict[int, str]:
    """Deterministic rule-based names from segment medians (thresholds in ``profiling.naming``).

    Rules are evaluated in priority order and each name can be used once; unmatched segments fall
    back to ``"Segment <n>"``. The thresholds were set after inspecting this dataset's profiles, so
    on different data they should be re-reviewed.
    """
    t = cfg["profiling"]["naming"]
    names: dict[int, str] = {}
    used: set[str] = set()
    for _, r in summary.iterrows():
        pf, cf = r["PURCHASES_FREQUENCY_median"], r["CASH_ADVANCE_FREQUENCY_median"]
        full, util = r["PRC_FULL_PAYMENT_median"], r["BALANCE_TO_LIMIT_median"]
        name = f"Segment {int(r['cluster'])}"
        if cf >= t["cash_borrower"]["cash_advance_frequency_min"] and pf <= t["cash_borrower"]["purchases_frequency_max"]:
            name = "Cash-Advance Borrowers"
        elif full >= t["full_payer"]["prc_full_payment_min"] and pf >= t["full_payer"]["purchases_frequency_min"]:
            name = "Frequent Full-Payers"
        elif pf <= t["low_activity"]["purchases_frequency_max"] and util <= t["low_activity"]["balance_to_limit_max"] and cf <= t["low_activity"]["cash_advance_frequency_max"]:
            name = "Low-Activity Light Users"
        elif pf >= t["active"]["purchases_frequency_min"] and full < t["active"]["prc_full_payment_max"]:
            name = "Active Balance-Carriers"
        if name in used:
            name = f"{name} ({int(r['cluster'])})"
        used.add(name)
        names[int(r["cluster"])] = name
    return names


# Observed vs assumed vs action. Keyed by segment name; the assumptions are hypotheses to test, not findings.
RECOMMENDATIONS: dict[str, dict[str, str]] = {
    "Cash-Advance Borrowers": {
        "assumption": "Likely a higher-risk and higher-fee-income group, since cash advances usually carry fees and higher rates. "
        "This dataset has no fee, interest or delinquency fields, so this is NOT verified.",
        "action": "Offer a lower-rate balance-transfer or personal-loan product; nudge toward everyday purchases with card rewards; "
        "monitor utilisation for early-warning review (do not auto-restrict on this label alone).",
    },
    "Low-Activity Light Users": {
        "assumption": "Probably low revenue today and a possible attrition risk, but also the largest headroom for activation. Unverified.",
        "action": "Activation campaign: first-purchase bonuses, spend-threshold rewards, merchant partnerships; "
        "review whether the card is still the right product (annual-fee-free tier) before spending on acquisition-style offers.",
    },
    "Frequent Full-Payers": {
        "assumption": "Likely low interest income but high interchange (swipe-fee) value and low credit risk. Unverified - no revenue data.",
        "action": "Retention and loyalty: premium rewards tier, limit increases, installment-plan promotions for big-ticket items; "
        "avoid interest-based offers that do not match their behaviour.",
    },
    "Active Balance-Carriers": {
        "assumption": "Likely the main interest-income group, with moderate risk that depends on utilisation and repayment trends. Unverified.",
        "action": "Cross-sell installment plans and balance-management tools; reward on-time payments; "
        "watch the sub-group with payments below the minimum due for hardship outreach.",
    },
}


def describe_segment(row: pd.Series) -> str:
    """One-sentence factual description from a ``segment_summary`` row (observed behaviour only)."""
    return (
        f"{row['segment_name']} ({row['pct_customers']:.1f}% of customers): median balance {row['BALANCE_median']:,.0f}, "
        f"median purchases {row['PURCHASES_median']:,.0f} (purchase frequency {row['PURCHASES_FREQUENCY_median']:.2f}), "
        f"median cash advance {row['CASH_ADVANCE_median']:,.0f}, median credit limit {row['CREDIT_LIMIT_median']:,.0f}, "
        f"median payments {row['PAYMENTS_median']:,.0f}, {row['pct_full_payers_ge_50']:.0f}% pay >=50% of the time in full."
    )
