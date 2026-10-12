"""End-to-end pipeline: audit -> clean -> features -> preprocess -> cluster -> select -> profile -> report."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

from . import visuals as V
from .cleaning import clean_data
from .clustering import ModelSpec, build_specs, run_grid
from .config import PROJECT_ROOT, load_config
from .data import audit_columns, audit_consistency, file_sha256, load_raw, top_correlations
from .experiments import compare_preprocessing, imputation_sensitivity, make_matrix
from .features import ENGINEERED_FEATURES, add_engineered_features
from .logging_utils import get_logger, setup_logging
from .preprocessing import high_correlation_pairs, select_log_columns
from .profiling import (
    RECOMMENDATIONS, build_assignments, describe_segment, fit_pca, name_segments, profile_index, segment_summary, verify_alignment,
)
from .selection import add_stability, pick_finalists, rank_and_select, screen_eligibility

log = get_logger("pipeline")


def _dirs(cfg: dict[str, Any]) -> dict[str, Path]:
    d = {k: PROJECT_ROOT / cfg["paths"][k] for k in ("processed_dir", "outputs_dir", "figures_dir", "models_dir", "reports_dir")}
    for p in d.values():
        p.mkdir(parents=True, exist_ok=True)
    return d


def _jsonable(o: Any) -> Any:
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if np.isnan(o) else float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, float) and np.isnan(o):
        return None
    return o


def run(config_path: str | None = None, reuse_prep_comparison: bool = False) -> dict[str, Any]:
    """Execute the full pipeline and return a results dictionary (also saved as ``outputs/run_summary.json``)."""
    t_start = time.perf_counter()
    cfg = load_config(config_path)
    dirs = _dirs(cfg)
    setup_logging(PROJECT_ROOT / cfg["paths"]["log_file"])
    seed = cfg["project"]["seed"]
    id_col = cfg["data"]["id_column"]
    out, fig = dirs["outputs_dir"], dirs["figures_dir"]
    res: dict[str, Any] = {"seed": seed}

    # ------------------------------------------------------------------ 1. load + audit
    raw_path = PROJECT_ROOT / cfg["paths"]["raw_data"]
    sha_before = file_sha256(raw_path)
    raw = load_raw(raw_path, id_col, cfg["data"]["numeric_columns"])
    num_cols = cfg["data"]["numeric_columns"]
    audit = audit_columns(raw, id_col)
    audit.to_csv(out / "data_audit_columns.csv", index=False)
    res["consistency"] = audit_consistency(raw, cfg["cleaning"]["purchase_sum_tolerance"])
    top_correlations(raw, id_col, 0.6).to_csv(out / "data_audit_top_correlations_spearman.csv", index=False)
    V.plot_missing(raw, fig / "eda_01_missing_values.png")
    V.plot_missing_mechanism(raw, fig / "eda_02_min_payments_missing_mechanism.png")
    V.plot_distributions(raw, num_cols, fig / "eda_03_distributions_raw.png")
    V.plot_distributions(raw, num_cols, fig / "eda_04_distributions_log1p.png", log1p=True)
    V.plot_correlation(raw, num_cols, fig / "eda_05_correlation_spearman.png", "spearman", "Spearman correlation (raw data)")
    V.plot_outlier_boxplots(raw, num_cols, fig / "eda_06_outlier_boxplots.png")
    V.plot_purchase_redundancy(raw, fig / "eda_07_purchase_redundancy.png")
    V.plot_outlier_share(audit, len(raw), fig / "eda_09_outlier_share.png")

    # ------------------------------------------------------------------ 2. clean + engineer
    clean, cleaning_log = clean_data(raw, cfg)
    df = add_engineered_features(clean)
    df.to_csv(dirs["processed_dir"] / "customers_clean.csv", index=False)
    (out / "cleaning_log.json").write_text(json.dumps(_jsonable(cleaning_log), indent=2))
    res["cleaning_log"] = cleaning_log
    eng_cols = ["BALANCE_TO_LIMIT", "PAYMENT_TO_MINPAY_RATIO", "ONEOFF_SHARE", "AVG_PURCHASE_TICKET", "AVG_CASH_ADVANCE_TICKET", "PAYMENT_TO_BALANCE"]
    V.plot_correlation(df, [c for c in cfg["features"]["sets"]["engineered"]], fig / "eda_08_correlation_model_features.png", "spearman", "Spearman correlation of the model feature set (cleaned)")

    # ------------------------------------------------------------------ 3. preprocessing comparison
    prep_path = out / "preprocessing_comparison.csv"
    if reuse_prep_comparison and prep_path.exists():
        prep = pd.read_csv(prep_path)
        log.info("Reusing cached preprocessing comparison")
    else:
        prep = compare_preprocessing(df, cfg)
        prep.to_csv(prep_path, index=False)
    V.plot_preprocessing_comparison(prep, fig / "prep_01_preprocessing_comparison.png")

    sel_pre = cfg["preprocessing"]["selected"]
    feat_cols = cfg["features"]["sets"][sel_pre["feature_set"]]
    log_cols = select_log_columns(df[feat_cols], cfg["preprocessing"]["log_skew_threshold"]) if sel_pre["log"] else []
    Xt, pre = make_matrix(df, feat_cols, cfg, sel_pre["log"], sel_pre["scaler"])
    X = Xt.to_numpy()
    res["preprocessing"] = {**sel_pre, "n_features_in": len(feat_cols), "n_features_out": int(X.shape[1]), "log_columns": log_cols,
                            "dropped_by_variance": sorted(set(feat_cols) - set(Xt.columns))}
    high_correlation_pairs(Xt, cfg["preprocessing"]["high_corr_threshold"]).to_csv(out / "preprocessing_high_correlation_pairs.csv", index=False)
    pd.concat([df[[id_col]], Xt], axis=1).to_csv(dirs["processed_dir"] / "features_transformed.csv", index=False)
    V.plot_skewness(df, feat_cols, fig / "prep_02_skewness_before_after.png", log_cols)
    V.plot_transform_steps(df, ["BALANCE", "ONEOFF_PURCHASES", "CASH_ADVANCE", "PAYMENTS", "MINIMUM_PAYMENTS", "PURCHASES_FREQUENCY"], log_cols,
                           fig / "prep_03_transform_steps.png")
    V.plot_correlation(Xt, list(Xt.columns), fig / "prep_04_correlation_transformed.png", "pearson", "Pearson correlation after preprocessing (model input space)")
    log.info("Selected preprocessing %s -> X%s", sel_pre, X.shape)

    # ------------------------------------------------------------------ 4. clustering grid
    specs = build_specs(cfg, X)
    log.info("Running %d candidate models", len(specs))
    grid, labels_by_id = run_grid(specs, X, seed)
    grid = screen_eligibility(grid, cfg)
    grid.drop(columns=["params"]).assign(params=grid["params"].astype(str)).to_csv(out / "clustering_grid_all_models.csv", index=False)
    V.plot_elbow(grid, fig / "eval_01_kmeans_elbow.png")
    V.plot_metrics_by_k(grid, fig / "eval_02_metrics_by_k.png")
    V.plot_gmm_bic(grid, fig / "eval_03_gmm_bic_aic.png")
    V.plot_density_models(grid, fig / "eval_04_density_models.png")

    # ------------------------------------------------------------------ 5. stability + selection
    finalists = add_stability(pick_finalists(grid, cfg["stability"]["finalists_per_algorithm"]), X, cfg)
    table = rank_and_select(finalists, cfg)
    winner = table[table["in_pool"]].iloc[0]
    chosen_id = cfg["selection"].get("final_model_id") or winner["model_id"]
    pool_ids = set(table.loc[table["in_pool"], "model_id"])
    if chosen_id not in pool_ids:
        raise ValueError(f"final_model_id '{chosen_id}' is not an eligible+stable finalist. Pool: {sorted(pool_ids)}")
    overridden = chosen_id != winner["model_id"]
    table["selected"] = table["model_id"] == chosen_id
    keep = ["model_id", "algorithm", "n_clusters", "noise_pct", "min_cluster_pct", "max_cluster_pct", "silhouette", "davies_bouldin",
            "calinski_harabasz", "inertia", "bic", "aic", "stability_ari_mean", "stability_ari_std", "eligible", "stable", "in_pool",
            "mean_rank", "selection_rank", "selected", "reject_reason"]
    table[[c for c in keep if c in table.columns]].round(4).to_csv(out / "model_comparison.csv", index=False)
    V.plot_model_comparison(table, fig / "eval_05_finalist_comparison.png", chosen_id)
    res["selection"] = {"chosen_model_id": chosen_id, "composite_rank_winner": winner["model_id"], "manual_override": bool(overridden),
                        "override_reason": " ".join(str(cfg["selection"].get("override_reason", "")).split()) if overridden else "",
                        "n_candidates": int(len(grid)), "n_eligible": int(grid["eligible"].sum()), "n_finalists": int(len(table))}

    # ------------------------------------------------------------------ 6. final model
    spec = next(s for s in specs if s.model_id == chosen_id)
    est = spec.build(seed)
    labels = np.asarray(est.fit_predict(X)).astype(int)
    if not np.array_equal(labels, labels_by_id[chosen_id].astype(int)):
        log.warning("Refit labels differ from grid labels for %s (non-deterministic?)", chosen_id)
    if hasattr(est, "predict"):
        re_pred = est.predict(pre.transform(df[feat_cols]).to_numpy())
        res["repredict_matches_labels"] = bool(np.array_equal(re_pred, labels))
    pca, xy = fit_pca(X, seed)
    from sklearn.decomposition import PCA as _PCA

    pca_full = _PCA(random_state=seed).fit(X)
    xyz = pca_full.transform(X)[:, :3]
    res["pca_cumulative_variance"] = [round(float(v), 4) for v in np.cumsum(pca_full.explained_variance_ratio_)]
    V.plot_pca_scree(pca_full.explained_variance_ratio_, fig / "eval_06_pca_scree.png")
    res["pca_explained_variance"] = [float(v) for v in pca.explained_variance_ratio_]

    # ------------------------------------------------------------------ 7. profile
    summary0 = segment_summary(df, labels, {int(c): f"Segment {c}" for c in np.unique(labels)})
    names = name_segments(summary0, cfg)
    summary = segment_summary(df, labels, names)
    idx = profile_index(df, labels)
    assignments = build_assignments(df, labels, id_col, names, xy)
    res["alignment"] = verify_alignment(assignments, df, labels, id_col)
    if res["alignment"]["n_label_mismatches"] or res["alignment"]["n_unassigned"] or not res["alignment"]["same_id_set"]:
        raise RuntimeError(f"Customer/label alignment failed: {res['alignment']}")
    assignments[[id_col, "cluster", "segment_name", "pca_1", "pca_2"]].round(5).to_csv(out / "customer_assignments.csv", index=False)
    df.merge(assignments, on=id_col, how="left", validate="one_to_one").to_csv(out / "customers_with_segments.csv", index=False)
    summary.round(4).to_csv(out / "segment_summary.csv", index=False)
    idx.rename(index=names).to_csv(out / "segment_profile_index.csv")
    res["segments"] = {names[int(r["cluster"])]: {"cluster": int(r["cluster"]), "n": int(r["n_customers"]), "pct": float(r["pct_customers"]),
                       "description": describe_segment(r), **RECOMMENDATIONS.get(names[int(r["cluster"])], {})} for _, r in summary.iterrows()}

    heat_feats = ["BALANCE", "PURCHASES", "ONEOFF_PURCHASES", "INSTALLMENTS_PURCHASES", "CASH_ADVANCE", "PURCHASES_FREQUENCY", "CASH_ADVANCE_FREQUENCY",
                  "PURCHASES_TRX", "CREDIT_LIMIT", "PAYMENTS", "PRC_FULL_PAYMENT", "BALANCE_TO_LIMIT", "PAYMENT_TO_MINPAY_RATIO"]
    V.plot_cluster_sizes(summary, fig / "seg_01_cluster_sizes.png")
    V.plot_pca(xy, labels, names, pca.explained_variance_ratio_, fig / "seg_02_pca_segments.png")
    V.plot_pca_other_views(xyz, labels, names, pca_full.explained_variance_ratio_, fig / "seg_05_pca_other_views.png")
    V.plot_profile_heatmap(idx, names, fig / "seg_03_profile_index_heatmap.png", heat_feats)
    V.plot_segment_bars(summary, fig / "seg_04_segment_comparison.png")

    # ------------------------------------------------------------------ 8. robustness
    k = int(len(np.unique(labels)))
    rob: dict[str, Any] = {}
    if spec.algorithm == "kmeans":
        sens = imputation_sensitivity(raw, cfg, sel_pre["feature_set"], sel_pre["log"], sel_pre["scaler"], k)
        sens.to_csv(out / "imputation_sensitivity.csv", index=False)
        rob["imputation_sensitivity"] = sens.round(4).to_dict("records")
        alt = {}
        for fs in ("base_no_total", "base_with_total", "engineered_reduced"):
            Xa, _ = make_matrix(df, cfg["features"]["sets"][fs], cfg, sel_pre["log"], sel_pre["scaler"])
            la = KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(Xa.to_numpy())
            alt[fs] = float(adjusted_rand_score(labels, la))
        rob["feature_set_agreement_ari"] = alt
    res["robustness"] = rob

    # ------------------------------------------------------------------ 9. persist
    mp_info = cleaning_log.get("min_payments_imputed", {})
    fill_values = {
        "CREDIT_LIMIT": float(raw["CREDIT_LIMIT"].median()),
        "MINIMUM_PAYMENTS": {"strategy": cleaning_log["min_payments_strategy"], "value": float(mp_info.get("fill", raw["MINIMUM_PAYMENTS"].median()))},
    }
    input_cols = [c for c in cfg["data"]["numeric_columns"] if c not in ("PURCHASES", "TENURE")]
    input_ranges = {c: {"min": float(df[c].min()), "median": float(df[c].median()), "p99": float(df[c].quantile(0.99)), "max": float(df[c].max())} for c in input_cols}
    joblib.dump({"preprocessor": pre, "model": est, "feature_columns": feat_cols, "segment_names": names, "model_id": chosen_id, "seed": seed,
                 "fill_values": fill_values, "input_columns": input_cols, "input_ranges": input_ranges,
                 "proportion_columns": cfg["cleaning"]["proportion_columns"]},
                dirs["models_dir"] / "final_model.joblib")
    res["raw_data_sha256_before"], res["raw_data_sha256_after"] = sha_before, file_sha256(raw_path)
    res["raw_data_unchanged"] = sha_before == res["raw_data_sha256_after"]
    res["runtime_seconds"] = round(time.perf_counter() - t_start, 1)
    (out / "run_summary.json").write_text(json.dumps(_jsonable(res), indent=2))
    log.info("Pipeline finished in %.0fs", res["runtime_seconds"])
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description="Run the customer segmentation pipeline")
    ap.add_argument("--config", default=None, help="Path to a YAML config (default: configs/config.yaml)")
    ap.add_argument("--reuse-prep-comparison", action="store_true", help="Reuse outputs/preprocessing_comparison.csv if it exists (saves ~3.5 min)")
    args = ap.parse_args()
    run(args.config, args.reuse_prep_comparison)


if __name__ == "__main__":
    main()
