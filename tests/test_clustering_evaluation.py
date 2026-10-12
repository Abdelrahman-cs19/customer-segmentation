import numpy as np
import pandas as pd
import pytest

from segmentation.clustering import ModelSpec, ScipyAgglomerative, build_specs, kdistance_eps_candidates, run_grid
from segmentation.evaluation import add_composite_rank, bootstrap_stability, internal_metrics


@pytest.fixture(scope="module")
def blobs():
    rng = np.random.default_rng(0)
    centers = np.array([[0, 0, 0], [8, 8, 0], [0, 8, 8]])
    X = np.vstack([c + rng.normal(0, 0.7, (150, 3)) for c in centers])
    y = np.repeat([0, 1, 2], 150)
    return X, y


ALL_SPECS = [
    ModelSpec("kmeans", {"n_clusters": 3, "n_init": 3}),
    ModelSpec("minibatch_kmeans", {"n_clusters": 3, "n_init": 3, "batch_size": 128}),
    ModelSpec("agglomerative", {"n_clusters": 3, "linkage": "ward"}),
    ModelSpec("gmm", {"n_clusters": 3, "covariance_type": "full", "n_init": 2}),
    ModelSpec("dbscan", {"eps": 1.5, "min_samples": 5}),
    ModelSpec("hdbscan", {"min_cluster_size": 30, "min_samples": 5}),
]


@pytest.mark.parametrize("spec", ALL_SPECS, ids=lambda s: s.algorithm)
def test_every_algorithm_recovers_separated_blobs(blobs, spec):
    X, _ = blobs
    labels = spec.build(42).fit_predict(X)
    m = internal_metrics(X, labels)
    assert m["n_clusters"] == 3
    assert m["silhouette"] > 0.6


def test_unknown_algorithm_raises():
    with pytest.raises(ValueError):
        ModelSpec("spectral", {}).build(0)


def test_model_id_is_stable_and_unique():
    ids = {s.model_id for s in ALL_SPECS}
    assert len(ids) == len(ALL_SPECS)
    assert ModelSpec("kmeans", {"n_init": 3, "n_clusters": 3}).model_id == ModelSpec("kmeans", {"n_clusters": 3, "n_init": 3}).model_id


def test_scipy_agglomerative_cut_gives_requested_k(blobs):
    X, _ = blobs
    a = ScipyAgglomerative(method="ward").fit(X)
    for k in (2, 3, 5):
        assert len(np.unique(a.cut(k))) == k


def test_run_grid_returns_rows_and_aligned_labels(blobs):
    X, _ = blobs
    df, labels = run_grid(ALL_SPECS, X, 42)
    assert len(df) == len(ALL_SPECS) == len(labels)
    assert all(len(v) == len(X) for v in labels.values())
    assert df.loc[df.algorithm == "kmeans", "inertia"].notna().all()
    assert df.loc[df.algorithm == "gmm", "bic"].notna().all()


def test_run_grid_skips_failing_model(blobs):
    X, _ = blobs
    bad = ModelSpec("kmeans", {"n_clusters": 10_000, "n_init": 1})  # more clusters than samples
    df, labels = run_grid([bad, ALL_SPECS[0]], X, 42)
    assert len(df) == 1 and len(labels) == 1


def test_grid_is_reproducible_with_fixed_seed(blobs):
    X, _ = blobs
    _, a = run_grid(ALL_SPECS[:4], X, 7)
    _, b = run_grid(ALL_SPECS[:4], X, 7)
    for k in a:
        assert np.array_equal(a[k], b[k])


def test_internal_metrics_noise_handling(blobs):
    X, y = blobs
    labels = y.copy()
    labels[:30] = -1
    m = internal_metrics(X, labels)
    assert m["noise_pct"] == pytest.approx(30 / len(X) * 100)
    assert m["n_clusters"] == 3


def test_internal_metrics_single_cluster_is_nan(blobs):
    X, _ = blobs
    m = internal_metrics(X, np.zeros(len(X), dtype=int))
    assert np.isnan(m["silhouette"]) and m["n_clusters"] == 1


def test_bootstrap_stability_high_on_clear_structure_and_low_on_noise(blobs):
    from sklearn.cluster import KMeans

    X, _ = blobs
    hi = bootstrap_stability(lambda rs: KMeans(3, n_init=3, random_state=rs), X, n_rounds=4)
    assert hi["stability_ari_mean"] > 0.95
    noise = np.random.default_rng(0).normal(size=(600, 5))
    lo = bootstrap_stability(lambda rs: KMeans(8, n_init=1, random_state=rs), noise, n_rounds=4)
    assert lo["stability_ari_mean"] < hi["stability_ari_mean"]


def test_composite_rank_prefers_better_model():
    d = pd.DataFrame({"silhouette": [0.5, 0.2], "davies_bouldin": [0.8, 1.9], "calinski_harabasz": [900, 300], "stability_ari_mean": [0.99, 0.5]})
    r = add_composite_rank(d)
    assert r.loc[0, "mean_rank"] < r.loc[1, "mean_rank"]


def test_kdistance_eps_candidates_sorted_positive(blobs):
    X, _ = blobs
    eps = kdistance_eps_candidates(X, 5, [0.8, 0.9, 0.95])
    assert eps == sorted(eps) and all(e > 0 for e in eps)


def test_build_specs_covers_all_algorithms(cfg, blobs):
    X, _ = blobs
    specs = build_specs(cfg, X)
    assert {s.algorithm for s in specs} == {"kmeans", "minibatch_kmeans", "agglomerative", "gmm", "dbscan", "hdbscan"}
