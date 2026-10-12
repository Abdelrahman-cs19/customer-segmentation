"""Eligibility screening, finalist stability testing and model selection."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .clustering import ModelSpec
from .evaluation import add_composite_rank, bootstrap_stability
from .logging_utils import get_logger

log = get_logger("selection")


def screen_eligibility(grid: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    """Flag each grid row with ``eligible`` and a human-readable ``reject_reason``.

    Hard constraints (from ``selection`` in the config): number of clusters inside
    [k_min, k_max], smallest cluster >= min_cluster_fraction, noise <= max_noise_fraction.
    These encode business usability, not statistical quality: a 1-customer segment or a 50%
    "noise" bucket cannot be marketed to, whatever its silhouette.
    """
    s = cfg["selection"]
    out = grid.copy()
    reasons = []
    for _, r in out.iterrows():
        why = []
        if not (s["k_min"] <= r["n_clusters"] <= s["k_max"]):
            why.append(f"k={int(r['n_clusters'])} outside [{s['k_min']},{s['k_max']}]")
        if not (r["min_cluster_pct"] >= s["min_cluster_fraction"] * 100):  # NaN-safe
            why.append(f"smallest cluster {r['min_cluster_pct']:.3g}% < {s['min_cluster_fraction'] * 100:g}%")
        if r["noise_pct"] > s["max_noise_fraction"] * 100:
            why.append(f"noise {r['noise_pct']:.1f}% > {s['max_noise_fraction'] * 100:g}%")
        reasons.append("; ".join(why))
    out["reject_reason"] = reasons
    out["eligible"] = out["reject_reason"].eq("")
    return out


def pick_finalists(screened: pd.DataFrame, per_algorithm: int) -> pd.DataFrame:
    """Pick finalists for the (expensive) stability test.

    For each algorithm: the top ``per_algorithm`` *eligible* rows by mean rank of
    silhouette/Davies-Bouldin/Calinski-Harabasz, plus - if no eligible row exists - the best
    ineligible row, so the final comparison table can show *why* an algorithm was rejected.
    """
    ranked = screened.copy()
    ranked["_r_sil"] = ranked["silhouette"].rank(ascending=False)
    ranked["_r_db"] = ranked["davies_bouldin"].rank(ascending=True)
    ranked["_r_ch"] = ranked["calinski_harabasz"].rank(ascending=False)
    ranked["_score3"] = ranked[["_r_sil", "_r_db", "_r_ch"]].mean(axis=1, skipna=False)
    picks = []
    for algo, grp in ranked.groupby("algorithm", sort=False):
        valid = grp.dropna(subset=["_score3"])
        elig = valid[valid["eligible"]].nsmallest(per_algorithm, "_score3")
        picks.append(elig if len(elig) else valid.nsmallest(1, "_score3"))
    return pd.concat(picks).drop(columns=["_r_sil", "_r_db", "_r_ch", "_score3"])


def add_stability(finalists: pd.DataFrame, X: np.ndarray, cfg: dict[str, Any]) -> pd.DataFrame:
    """Attach bootstrap stability (mean/std ARI) to every finalist."""
    st = cfg["stability"]
    seed = cfg["project"]["seed"]
    out = finalists.copy()
    means, stds = [], []
    for _, r in out.iterrows():
        spec = ModelSpec(r["algorithm"], dict(r["params"]))
        res = bootstrap_stability(spec.build, X, n_rounds=st["n_rounds"], fraction=st["subsample_fraction"], seed=seed)
        means.append(res["stability_ari_mean"])
        stds.append(res["stability_ari_std"])
        log.info("stability %-70s ARI=%.3f +/- %.3f", r["model_id"], res["stability_ari_mean"], res["stability_ari_std"])
    out["stability_ari_mean"], out["stability_ari_std"] = means, stds
    return out


def rank_and_select(finalists: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    """Rank eligible+stable finalists by equal-weight composite rank and mark the shortlist.

    Adds ``stable``, ``in_pool``, ``mean_rank`` (NaN outside the pool), and ``selection_rank``.
    The composite rank is deliberately *a tool for shortlisting*; the final choice also weighs
    interpretability (see the report), and may be overridden via ``selection.final_model_id``.
    """
    s = cfg["selection"]
    out = finalists.copy()
    out["stable"] = out["stability_ari_mean"] >= s["min_stability_ari"]
    out["in_pool"] = out["eligible"] & out["stable"]
    ranked = add_composite_rank(out[out["in_pool"]])
    out["mean_rank"] = np.nan
    out.loc[ranked.index, "mean_rank"] = ranked["mean_rank"]
    out["selection_rank"] = out["mean_rank"].rank(method="min")
    if (not out["eligible"].all()) or (not out["stable"].all()):
        reasons = []
        for _, r in out.iterrows():
            why = r["reject_reason"]
            if r["eligible"] and not r["stable"]:
                why = f"stability ARI {r['stability_ari_mean']:.2f} < {s['min_stability_ari']:g}"
            reasons.append(why)
        out["reject_reason"] = reasons
    return out.sort_values(["in_pool", "mean_rank"], ascending=[False, True]).reset_index(drop=True)
