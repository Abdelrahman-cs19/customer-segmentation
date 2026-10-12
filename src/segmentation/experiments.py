"""Pre-modelling experiments: feature-set x preprocessing comparison and imputation sensitivity."""
from __future__ import annotations

import time
from itertools import product
from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

from .cleaning import clean_data
from .evaluation import bootstrap_stability, internal_metrics
from .features import add_engineered_features
from .logging_utils import get_logger
from .preprocessing import build_preprocessor, select_log_columns, validate_transformed

log = get_logger("experiments")


def make_matrix(df: pd.DataFrame, columns: list[str], cfg: dict[str, Any], log_on: bool, scaler: str) -> tuple[pd.DataFrame, Any]:
    """Fit the preprocessing pipeline on ``df[columns]`` and return (validated matrix, fitted pipeline)."""
    pcfg = cfg["preprocessing"]
    log_cols = select_log_columns(df[columns], pcfg["log_skew_threshold"]) if log_on else []
    pre = build_preprocessor(log_cols, scaler, pcfg["variance_threshold"])
    Xt = pre.fit_transform(df[columns])
    validate_transformed(Xt, len(df))
    return Xt, pre


def compare_preprocessing(df: pd.DataFrame, cfg: dict[str, Any], ks: tuple[int, ...] = (3, 4, 5, 6, 7, 8), n_rounds: int = 5) -> pd.DataFrame:
    """Evaluate every (feature set, log, scaler) combination with K-Means over several ``k``.

    Caveat recorded in the report: internal metrics computed in *different* feature spaces are not
    strictly comparable (e.g. silhouette in a 19-D space vs a 17-D space). The most comparable
    signals are stability (ARI), smallest-cluster size and the interpretability review.
    """
    seed = cfg["project"]["seed"]
    rows = []
    variants = cfg["preprocessing"]["variants"]
    for set_name, v in product(cfg["features"]["sets"], variants):
        cols = cfg["features"]["sets"][set_name]
        t0 = time.perf_counter()
        Xt, _ = make_matrix(df, cols, cfg, v["log"], v["scaler"])
        X = Xt.to_numpy()
        for k in ks:
            labels = KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(X)
            m = internal_metrics(X, labels, seed)
            stab = bootstrap_stability(lambda rs, k=k: KMeans(n_clusters=k, n_init=3, random_state=rs), X, n_rounds=n_rounds, seed=seed)
            rows.append({"feature_set": set_name, "log1p": v["log"], "scaler": v["scaler"], "k": k, "n_features": X.shape[1], **m, **stab})
        log.info("preprocessing variant %s | log=%s | %s done in %.1fs", set_name, v["log"], v["scaler"], time.perf_counter() - t0)
    return pd.DataFrame(rows)


def imputation_sensitivity(raw: pd.DataFrame, cfg: dict[str, Any], feature_set: str, log_on: bool, scaler: str, k: int) -> pd.DataFrame:
    """Re-cluster with each MINIMUM_PAYMENTS strategy and compare to the 'median' baseline via ARI.

    High ARI means the imputation choice does not materially change the segmentation.
    """
    seed = cfg["project"]["seed"]
    cols = cfg["features"]["sets"][feature_set]
    results: dict[str, np.ndarray] = {}
    for strat in ("median", "zero", "balance_ratio"):
        clean, _ = clean_data(raw, cfg, min_payments_strategy=strat)
        df = add_engineered_features(clean)
        Xt, _ = make_matrix(df, cols, cfg, log_on, scaler)
        results[strat] = KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(Xt.to_numpy())
    base = results["median"]
    return pd.DataFrame(
        [{"strategy": s, "ari_vs_median": adjusted_rand_score(base, lab), "n_clusters": len(np.unique(lab))} for s, lab in results.items()]
    )
