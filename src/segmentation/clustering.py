"""Clustering model specifications and grid execution."""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.cluster import DBSCAN, HDBSCAN, KMeans, MiniBatchKMeans
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors

from .evaluation import internal_metrics
from .logging_utils import get_logger

log = get_logger("clustering")


class ScipyAgglomerative(ClusterMixin, BaseEstimator):
    """Agglomerative clustering via SciPy, so one dendrogram can be cut at many ``k``.

    Equivalent to sklearn's ``AgglomerativeClustering`` for Euclidean ward/average/complete linkage.
    """

    def __init__(self, n_clusters: int = 3, method: str = "ward"):
        self.n_clusters = n_clusters
        self.method = method

    def fit(self, X, y=None):
        self.Z_ = linkage(np.asarray(X), method=self.method)
        self.labels_ = self.cut(self.n_clusters)
        return self

    def cut(self, k: int) -> np.ndarray:
        """Cut the fitted dendrogram into ``k`` clusters (labels start at 0)."""
        return fcluster(self.Z_, t=k, criterion="maxclust") - 1


class GMMClusterer(ClusterMixin, BaseEstimator):
    """Gaussian Mixture wrapper exposing ``fit_predict`` and keeping BIC/AIC."""

    def __init__(self, n_components: int = 3, covariance_type: str = "full", n_init: int = 3, random_state: int = 42):
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.n_init = n_init
        self.random_state = random_state

    def fit(self, X, y=None):
        self.gmm_ = GaussianMixture(
            n_components=self.n_components,
            covariance_type=self.covariance_type,
            n_init=self.n_init,
            random_state=self.random_state,
            reg_covar=1e-5,
            max_iter=300,
        ).fit(X)
        self.labels_ = self.gmm_.predict(X)
        self.bic_ = float(self.gmm_.bic(X))
        self.aic_ = float(self.gmm_.aic(X))
        return self

    def predict(self, X):
        return self.gmm_.predict(X)


@dataclass
class ModelSpec:
    """A single (algorithm, hyper-parameters) candidate."""

    algorithm: str
    params: dict[str, Any] = field(default_factory=dict)

    @property
    def model_id(self) -> str:
        p = ",".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in sorted(self.params.items()))
        return f"{self.algorithm}[{p}]"

    def build(self, seed: int):
        """Instantiate an unfitted estimator with a fixed seed."""
        p = dict(self.params)
        if self.algorithm == "kmeans":
            return KMeans(n_clusters=p["n_clusters"], n_init=p["n_init"], random_state=seed)
        if self.algorithm == "minibatch_kmeans":
            return MiniBatchKMeans(n_clusters=p["n_clusters"], n_init=p["n_init"], batch_size=p["batch_size"], random_state=seed)
        if self.algorithm == "agglomerative":
            return ScipyAgglomerative(n_clusters=p["n_clusters"], method=p["linkage"])
        if self.algorithm == "gmm":
            return GMMClusterer(p["n_clusters"], p["covariance_type"], p["n_init"], seed)
        if self.algorithm == "dbscan":
            return DBSCAN(eps=p["eps"], min_samples=p["min_samples"])
        if self.algorithm == "hdbscan":
            return HDBSCAN(min_cluster_size=p["min_cluster_size"], min_samples=p["min_samples"], copy=True)
        raise ValueError(f"Unknown algorithm: {self.algorithm}")


def kdistance_eps_candidates(X: np.ndarray, min_samples: int, quantiles: list[float]) -> list[float]:
    """Candidate DBSCAN ``eps`` values from quantiles of the k-distance curve (k = min_samples)."""
    nn = NearestNeighbors(n_neighbors=min_samples).fit(X)
    dist, _ = nn.kneighbors(X)
    kd = np.sort(dist[:, -1])
    return sorted({round(float(np.quantile(kd, q)), 4) for q in quantiles})


def build_specs(cfg: dict[str, Any], X: np.ndarray) -> list[ModelSpec]:
    """Expand the config into the full list of candidate models."""
    c = cfg["clustering"]
    k_lo, k_hi = c["k_range"]
    ks = range(k_lo, k_hi + 1)
    specs: list[ModelSpec] = []
    for k in ks:
        specs.append(ModelSpec("kmeans", {"n_clusters": k, **c["kmeans"]}))
    for k in ks:
        specs.append(ModelSpec("minibatch_kmeans", {"n_clusters": k, **c["minibatch_kmeans"]}))
    for link in c["agglomerative"]["linkages"]:
        for k in ks:
            specs.append(ModelSpec("agglomerative", {"n_clusters": k, "linkage": link}))
    for cov in c["gmm"]["covariance_types"]:
        for k in ks:
            specs.append(ModelSpec("gmm", {"n_clusters": k, "covariance_type": cov, "n_init": c["gmm"]["n_init"]}))
    for ms in c["dbscan"]["min_samples"]:
        for eps in kdistance_eps_candidates(X, ms, c["dbscan"]["eps_quantiles"]):
            specs.append(ModelSpec("dbscan", {"eps": eps, "min_samples": ms}))
    for mcs in c["hdbscan"]["min_cluster_size"]:
        for ms in c["hdbscan"]["min_samples"]:
            specs.append(ModelSpec("hdbscan", {"min_cluster_size": mcs, "min_samples": ms}))
    return specs


def run_grid(specs: list[ModelSpec], X: np.ndarray, seed: int) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    """Fit and evaluate every spec. A failing model is logged and skipped, never fatal.

    Agglomerative dendrograms are computed once per linkage and cut for each ``k``.

    Returns:
        (results dataframe, {model_id: labels})
    """
    rows: list[dict[str, Any]] = []
    labels_by_id: dict[str, np.ndarray] = {}
    agglo_cache: dict[str, ScipyAgglomerative] = {}
    agglo_time: dict[str, float] = {}

    for i, spec in enumerate(specs, 1):
        t0 = time.perf_counter()
        extra: dict[str, Any] = {}
        try:
            if spec.algorithm == "agglomerative":
                link = spec.params["linkage"]
                if link not in agglo_cache:
                    t_link = time.perf_counter()
                    agglo_cache[link] = ScipyAgglomerative(method=link).fit(X)
                    agglo_time[link] = time.perf_counter() - t_link
                labels = agglo_cache[link].cut(spec.params["n_clusters"])
                fit_seconds = agglo_time[link]
            else:
                est = spec.build(seed)
                labels = est.fit_predict(X)
                fit_seconds = time.perf_counter() - t0
                if hasattr(est, "inertia_"):
                    extra["inertia"] = float(est.inertia_)
                if hasattr(est, "bic_"):
                    extra["bic"], extra["aic"] = est.bic_, est.aic_
            metrics = internal_metrics(X, labels, seed)
        except Exception as exc:  # noqa: BLE001 - one bad config must not abort the grid
            log.warning("Model %s failed: %s", spec.model_id, exc)
            continue
        rows.append(
            {
                "model_id": spec.model_id,
                "algorithm": spec.algorithm,
                "params": spec.params,
                "k_param": spec.params.get("n_clusters", np.nan),
                "fit_seconds": round(fit_seconds, 3),
                **metrics,
                **extra,
            }
        )
        labels_by_id[spec.model_id] = np.asarray(labels)
        if i % 20 == 0 or i == len(specs):
            log.info("  grid progress %d/%d", i, len(specs))
    return pd.DataFrame(rows), labels_by_id
