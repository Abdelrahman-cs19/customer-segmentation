"""Internal validation metrics, stability testing and model ranking."""
from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score, calinski_harabasz_score, davies_bouldin_score, silhouette_score

NOISE_LABEL = -1


def internal_metrics(X: np.ndarray, labels: np.ndarray, seed: int = 42) -> dict[str, float]:
    """Compute internal validity metrics.

    Noise points (label -1, from DBSCAN/HDBSCAN) are excluded from the metrics and reported
    separately via ``noise_pct``. Metrics are NaN when fewer than 2 clusters remain.
    """
    labels = np.asarray(labels)
    noise = labels == NOISE_LABEL
    core = ~noise
    clusters, counts = np.unique(labels[core], return_counts=True)
    res = {
        "n_clusters": int(len(clusters)),
        "noise_pct": float(noise.mean() * 100),
        "min_cluster_pct": float(counts.min() / len(labels) * 100) if len(counts) else np.nan,
        "max_cluster_pct": float(counts.max() / len(labels) * 100) if len(counts) else np.nan,
        "silhouette": np.nan,
        "davies_bouldin": np.nan,
        "calinski_harabasz": np.nan,
    }
    if len(clusters) >= 2 and core.sum() > len(clusters):
        Xc, lc = X[core], labels[core]
        res["silhouette"] = float(silhouette_score(Xc, lc, random_state=seed))
        res["davies_bouldin"] = float(davies_bouldin_score(Xc, lc))
        res["calinski_harabasz"] = float(calinski_harabasz_score(Xc, lc))
    return res


def bootstrap_stability(
    factory: Callable[[int], object],
    X: np.ndarray,
    n_rounds: int = 10,
    fraction: float = 0.8,
    seed: int = 42,
) -> dict[str, float]:
    """Subsample-based stability: mean Adjusted Rand Index between independent refits.

    Each round draws two random subsamples (``fraction`` of rows, without replacement), refits
    the model on each, and compares the two labelings on the rows they share. ARI near 1 means
    the partition is reproducible; near 0 means it is largely an artefact of the sample.

    Args:
        factory: ``factory(random_state) -> estimator`` exposing ``fit_predict``.
    """
    rng = np.random.default_rng(seed)
    n = len(X)
    m = int(n * fraction)
    scores = []
    for r in range(n_rounds):
        idx_a = np.sort(rng.choice(n, m, replace=False))
        idx_b = np.sort(rng.choice(n, m, replace=False))
        lab_a = pd.Series(factory(seed + 2 * r).fit_predict(X[idx_a]), index=idx_a)
        lab_b = pd.Series(factory(seed + 2 * r + 1).fit_predict(X[idx_b]), index=idx_b)
        common = lab_a.index.intersection(lab_b.index)
        scores.append(adjusted_rand_score(lab_a.loc[common], lab_b.loc[common]))
    return {"stability_ari_mean": float(np.mean(scores)), "stability_ari_std": float(np.std(scores))}


def add_composite_rank(df: pd.DataFrame) -> pd.DataFrame:
    """Add an equal-weight mean rank across silhouette (high), Davies-Bouldin (low),
    Calinski-Harabasz (high) and stability (high). Lower mean rank is better.

    Rows with missing values in any ranked metric get NaN. Ranking is relative to ``df``, so call
    it on the exact candidate pool being compared.
    """
    out = df.copy()
    ranks = pd.DataFrame(
        {
            "r_sil": out["silhouette"].rank(ascending=False),
            "r_db": out["davies_bouldin"].rank(ascending=True),
            "r_ch": out["calinski_harabasz"].rank(ascending=False),
            "r_stab": out["stability_ari_mean"].rank(ascending=False),
        }
    )
    out["mean_rank"] = ranks.mean(axis=1, skipna=False)
    return out
