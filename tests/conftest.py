"""Shared fixtures: a small synthetic frame with the real schema, missing values and invalid entries."""
import numpy as np
import pandas as pd
import pytest

from segmentation.config import load_config


@pytest.fixture(scope="session")
def cfg():
    return load_config()


@pytest.fixture()
def raw_df(cfg) -> pd.DataFrame:
    """300 synthetic customers with deliberate defects: NaNs, a frequency > 1, zero payments."""
    rng = np.random.default_rng(0)
    n = 300
    oneoff = np.where(rng.random(n) < 0.4, 0, rng.gamma(2, 300, n))
    inst = np.where(rng.random(n) < 0.4, 0, rng.gamma(2, 200, n))
    cash = np.where(rng.random(n) < 0.5, 0, rng.gamma(2, 500, n))
    df = pd.DataFrame(
        {
            "CUST_ID": [f"C{i:05d}" for i in range(n)],
            "BALANCE": rng.gamma(2, 800, n),
            "BALANCE_FREQUENCY": rng.choice([0.5, 0.9, 1.0], n),
            "PURCHASES": oneoff + inst,
            "ONEOFF_PURCHASES": oneoff,
            "INSTALLMENTS_PURCHASES": inst,
            "CASH_ADVANCE": cash,
            "PURCHASES_FREQUENCY": rng.random(n),
            "ONEOFF_PURCHASES_FREQUENCY": rng.random(n) * 0.5,
            "PURCHASES_INSTALLMENTS_FREQUENCY": rng.random(n) * 0.8,
            "CASH_ADVANCE_FREQUENCY": np.where(cash > 0, rng.random(n) * 0.5, 0.0),
            "CASH_ADVANCE_TRX": np.where(cash > 0, rng.integers(1, 10, n), 0),
            "PURCHASES_TRX": np.where(oneoff + inst > 0, rng.integers(1, 40, n), 0),
            "CREDIT_LIMIT": rng.choice([1000.0, 3000.0, 5000.0, 9000.0], n),
            "PAYMENTS": rng.gamma(2, 600, n),
            "MINIMUM_PAYMENTS": rng.gamma(2, 200, n) + 1,
            "PRC_FULL_PAYMENT": rng.choice([0.0, 0.0, 0.5, 1.0], n),
            "TENURE": rng.choice([6, 12, 12, 12], n),
        }
    )
    df.loc[[3, 4, 5], "MINIMUM_PAYMENTS"] = np.nan
    df.loc[[3, 4], "PAYMENTS"] = 0.0
    df.loc[7, "CREDIT_LIMIT"] = np.nan
    df.loc[9, "CASH_ADVANCE_FREQUENCY"] = 1.4
    return df
