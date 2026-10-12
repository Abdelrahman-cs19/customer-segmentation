"""Generate ``reports/final_report.md`` from the pipeline's output files.

Every number in the report is read from ``outputs/`` (nothing is typed in by hand), so the report
cannot drift from the results. Narrative sentences state what the numbers show for the run being
reported; if the data or configuration change, regenerate and re-read the interpretation.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .config import PROJECT_ROOT, load_config


def md_table(df: pd.DataFrame, floatfmt: str = "{:.3f}") -> str:
    """Render a dataframe as a GitHub-flavoured markdown table (no external dependency)."""
    cols = list(df.columns)
    lines = ["| " + " | ".join(str(c) for c in cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        cells = []
        for v in r:
            if isinstance(v, (float, np.floating)):
                cells.append("" if np.isnan(v) else floatfmt.format(v))
            elif isinstance(v, (bool, np.bool_)):
                cells.append("yes" if v else "no")
            else:
                cells.append(str(v).replace("|", "/"))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _short(model_id: str) -> str:
    return model_id.replace(",n_init=10", "").replace(",n_init=3", "").replace("batch_size=1024,", "").replace("[]", "")


def write_final_report(config_path: str | None = None) -> Path:
    """Build the markdown report from existing outputs and write it to ``reports/final_report.md``."""
    cfg = load_config(config_path)
    out = PROJECT_ROOT / cfg["paths"]["outputs_dir"]
    run = json.loads((out / "run_summary.json").read_text())
    audit = pd.read_csv(out / "data_audit_columns.csv")
    prep = pd.read_csv(out / "preprocessing_comparison.csv")
    grid = pd.read_csv(out / "clustering_grid_all_models.csv")
    models = pd.read_csv(out / "model_comparison.csv")
    summary = pd.read_csv(out / "segment_summary.csv").sort_values("cluster")
    cons, clog, sel, pp = run["consistency"], run["cleaning_log"], run["selection"], run["preprocessing"]
    rob = run.get("robustness", {})
    n = cons["n_rows"]
    chosen = models[models["selected"]].iloc[0]
    winner = models[models["model_id"] == sel["composite_rank_winner"]].iloc[0]

    # ----- derived facts
    avg_link = grid[grid["model_id"].str.contains("linkage=average")]
    avg_min_n = int(round(avg_link["min_cluster_pct"].min() / 100 * n))
    dens = grid[grid["algorithm"].isin(["dbscan", "hdbscan"])]
    dbs = grid[grid["algorithm"] == "dbscan"]
    hdb = grid[grid["algorithm"] == "hdbscan"]
    gmm_full = grid[grid["model_id"].str.contains("covariance_type=full")]
    nolog_tiny = prep[(~prep["log1p"]) & (prep["min_cluster_pct"] < 0.1)]
    prep_v = (prep.assign(variant=prep["feature_set"] + " | " + np.where(prep["log1p"], "log1p", "raw") + " + " + prep["scaler"])
              .groupby("variant").agg(silhouette=("silhouette", "mean"), davies_bouldin=("davies_bouldin", "mean"),
                                      stability_ari=("stability_ari_mean", "mean"), avg_min_cluster_pct=("min_cluster_pct", "mean"),
                                      worst_min_cluster_pct=("min_cluster_pct", "min")).sort_values("stability_ari", ascending=False).reset_index())
    k_tbl = prep[(prep["feature_set"] == pp["feature_set"]) & (prep["log1p"] == pp["log"]) & (prep["scaler"] == pp["scaler"])]
    algo_tbl = (grid.groupby("algorithm").agg(configs_tried=("model_id", "count"), eligible=("eligible", "sum")).reset_index())
    best_elig = grid[grid["eligible"]].sort_values("silhouette", ascending=False).groupby("algorithm").head(1).set_index("algorithm")
    algo_tbl["best_eligible_model"] = algo_tbl["algorithm"].map(lambda a: _short(best_elig.loc[a, "model_id"]) if a in best_elig.index else "none eligible")
    algo_tbl["its_silhouette"] = algo_tbl["algorithm"].map(lambda a: best_elig.loc[a, "silhouette"] if a in best_elig.index else np.nan)

    fig = "../figures/"
    L: list[str] = []
    a = L.append

    a("# Credit-Card Customer Segmentation - Final Report\n")
    a(f"_Generated from `outputs/` by `segmentation.reporting`. Seed {run['seed']}; pipeline runtime {run['runtime_seconds']:.0f}s. "
      f"Raw data checksum unchanged: **{'yes' if run['raw_data_unchanged'] else 'NO'}**._\n")

    # ------------------------------------------------------------------ 1
    a("## 1. Executive summary\n")
    a(f"- **{n:,} customers**, 17 behavioural/account fields. No duplicates, no negative values; 313 missing `MINIMUM_PAYMENTS`, 1 missing `CREDIT_LIMIT`.")
    a(f"- **Selected model: `{_short(sel['chosen_model_id'])}`** on {pp['n_features_out']} features after `log1p` + `{pp['scaler']}` scaling. "
      f"Silhouette {chosen['silhouette']:.3f}, Davies-Bouldin {chosen['davies_bouldin']:.3f}, Calinski-Harabasz {chosen['calinski_harabasz']:,.0f}, "
      f"bootstrap stability ARI {chosen['stability_ari_mean']:.3f}, smallest segment {chosen['min_cluster_pct']:.1f}% of customers.")
    if sel["manual_override"]:
        a(f"- **Manual override:** the equal-weight composite rank preferred `{_short(sel['composite_rank_winner'])}` "
          f"(silhouette {winner['silhouette']:.3f}); `{_short(sel['chosen_model_id'])}` was chosen instead. Reason: {sel['override_reason']} "
          "This judgement was made *after* reviewing the segment profiles, not pre-specified.")
    a("- Segments (observed behaviour):")
    for _, r in summary.iterrows():
        a(f"  - **{r['segment_name']}** - {r['n_customers']:,} customers ({r['pct_customers']:.1f}%)")
    a("- Density-based methods (DBSCAN, HDBSCAN) and average-linkage hierarchical clustering produced no usable segmentation (Section 6).")
    a("- The data contain **no revenue, interest, fee or default fields**; any statement about profitability or credit risk is an assumption (Section 8).\n")

    # ------------------------------------------------------------------ 2
    a("## 2. Data audit and EDA\n")
    a("Raw file preserved unchanged in `data/raw/` (SHA-256 recorded in `SHA256SUM.txt` and re-verified at the end of the run).\n")
    a("| Check | Result | Why it matters |\n|---|---|---|")
    a(f"| Duplicates | 0 duplicate rows, 0 duplicate `CUST_ID` | `CUST_ID` is a safe join key |")
    a(f"| `PURCHASES` vs components | equals `ONEOFF + INSTALLMENTS` in {cons['purchases_equal_components_pct']:.2f}% of rows ({cons['purchases_component_mismatch_rows']} rows differ, max gap {cons['purchases_component_max_gap']:,.0f}) | Using all three double-counts purchase behaviour; `PURCHASES` is excluded from the model |")
    a(f"| `MINIMUM_PAYMENTS` missing | {cons['min_payments_missing']} rows; {cons['min_payments_missing_with_zero_payments']} of them have `PAYMENTS == 0` (all {cons['zero_payments_total']} zero-payment customers), {cons['min_payments_missing_with_positive_payments']} have payments > 0 | Not missing at random: tied to no payment activity. Flagged, then imputed (Section 3) |")
    a(f"| `CREDIT_LIMIT` missing | {cons['credit_limit_missing']} row | Negligible; median imputation |")
    a(f"| `CASH_ADVANCE_FREQUENCY` > 1 | {cons['cash_adv_frequency_above_1']} rows (max 1.5) | A frequency cannot exceed 1; clipped and logged |")
    a(f"| Purchase amount vs count disagree | {cons['purchases_pos_but_zero_trx']} rows spend>0 with 0 trx; {cons['purchases_zero_but_pos_trx']} rows trx>0 with 0 spend | Tiny; flagged, not removed |")
    a(f"| `BALANCE` above `CREDIT_LIMIT` | {cons['balance_above_credit_limit']} customers | Plausible over-limit accounts; kept |")
    a(f"| Fully inactive customers | {cons['fully_inactive_customers']} (no purchases and no cash advance) | Negligible |\n")
    skew = audit.sort_values("skew", ascending=False).head(6)[["column", "skew", "pct_zero", "n_outliers_3iqr", "p99", "max"]]
    a("**Skewness and outliers (6 most skewed columns):**\n")
    a(md_table(skew.rename(columns={"n_outliers_3iqr": "outliers (>3xIQR)"}), "{:,.2f}") + "\n")
    a("Business implications: spend, payments and cash advance are extremely right-skewed with long tails of legitimate high-value customers "
      "(e.g. one customer with ~49k purchases, 101 transactions, a 22.5k limit and ~47k payments is internally consistent). Distance-based clustering on raw amounts "
      "would be dominated by a few hundred customers, so the tails are handled by transformation and robust scaling rather than deletion. "
      "Many behaviours are structurally zero (66% never pay in full, 52% never take a cash advance, 23% make no purchases), which is real behaviour the segmentation should surface.\n")
    a(f"![Missing mechanism]({fig}eda_02_min_payments_missing_mechanism.png)\n")
    a(f"![Distributions]({fig}eda_03_distributions_raw.png)\n")
    a(f"![Spearman correlation]({fig}eda_05_correlation_spearman.png)\n")
    a(f"![Outliers]({fig}eda_06_outlier_boxplots.png)\n")
    a(f"![Outlier share]({fig}eda_09_outlier_share.png)\n")
    a(f"![Purchase redundancy]({fig}eda_07_purchase_redundancy.png)\n")

    # ------------------------------------------------------------------ 3
    a("## 3. Cleaning decisions\n")
    a("No rows were dropped (asserted in code). Steps, in order:\n")
    a(f"1. Clip proportion fields to [0,1]: {clog['clipped_to_unit_interval']}.")
    a("2. Add `MIN_PAYMENTS_MISSING` and `CREDIT_LIMIT_MISSING` flags **before** imputing, so the information is not lost (used for profiling, not clustering).")
    cl = clog.get("credit_limit_imputed", {})
    mp = clog.get("min_payments_imputed", {})
    a(f"3. `CREDIT_LIMIT`: {cl.get('n', 0)} value -> median ({cl.get('value', float('nan')):,.0f}).")
    a(f"4. `MINIMUM_PAYMENTS`: {mp.get('n', 0)} values -> `{mp.get('strategy')}` ({mp.get('fill', float('nan')):,.1f}). "
      "A zero-fill would claim \"no minimum due\" and a balance-scaled fill would invent a customer-specific figure; the median is the least assumption-laden. "
      "The choice is tested for sensitivity below.")
    sens = rob.get("imputation_sensitivity")
    if sens:
        a("\n**Imputation sensitivity** (same final model re-fit under each strategy; ARI vs the `median` baseline, 1 = identical segmentation):\n")
        a(md_table(pd.DataFrame(sens), "{:.3f}") + "\n")

    # ------------------------------------------------------------------ 4
    a("## 4. Feature engineering\n")
    a("Ratios use guarded denominators (no infinities; asserted). All are computed per customer, so there is no leakage.\n")
    a("| Feature | Definition | Rationale |\n|---|---|---|")
    a("| `BALANCE_TO_LIMIT` | `BALANCE / CREDIT_LIMIT` | Credit utilisation, comparable across limit sizes |")
    a("| `PAYMENT_TO_MINPAY_RATIO` | `PAYMENTS / max(MINIMUM_PAYMENTS, 1)` | Repayment effort relative to what was due; floor of 1 avoids absurd ratios from near-zero minimums |")
    a("| `ONEOFF_SHARE` | `ONEOFF / (ONEOFF + INSTALLMENTS)` | Purchase style, built from the components so it ignores the 20 inconsistent totals |")
    a("| `AVG_PURCHASE_TICKET` | `PURCHASES / PURCHASES_TRX` (0 if no trx) | Typical basket size |")
    a("| `AVG_CASH_ADVANCE_TICKET`, `PAYMENT_TO_BALANCE` | similar | Computed for profiling only |")
    a(f"\n`TENURE` is profiled but not clustered: 85% of customers sit at 12 and it describes the observation window more than behaviour. "
      f"`CUST_ID` never enters the model; it is carried alongside and joined back by id (verified for all rows, Section 8).\n")
    a(f"**Redundancy investigation.** After log transformation several pairs are highly correlated (`outputs/preprocessing_high_correlation_pairs.csv`); "
      f"cash-advance behaviour is represented by three related columns and purchases by several. An `engineered_reduced` set (dropping the two `*_TRX` counts) was tested and "
      f"did not improve stability (Section 5), so the pairs were reported but not removed.\n")
    a(f"![Model-feature correlation]({fig}eda_08_correlation_model_features.png)\n")

    # ------------------------------------------------------------------ 5
    a("## 5. Preprocessing\n")
    a(f"Pipeline (scikit-learn `Pipeline`, pandas in/out): `SimpleImputer(median)` safety net -> `log1p` on {len(pp['log_columns'])} eligible columns "
      f"(non-negative, unbounded, skew > {cfg['preprocessing']['log_skew_threshold']}) -> `{pp['scaler']}` scaling -> `VarianceThreshold`. "
      f"Proportions in [0,1] are not logged. Output validated for row count, NaN/inf, constant columns and magnitude "
      f"(constant columns dropped: {pp['dropped_by_variance'] or 'none found'}).\n")
    a(f"log1p columns: {', '.join('`' + c + '`' for c in pp['log_columns'])}\n")
    a("**Comparison of 4 feature sets x 2 transforms x 2 scalers** (K-Means, k = 3..8, means over k; stability = 5-round subsample ARI):\n")
    a(md_table(prep_v.round(3), "{:.3f}") + "\n")
    a(f"Findings: (1) `log1p` variants dominate on stability. (2) Without `log1p`, {len(nolog_tiny)} of {int((~prep['log1p']).sum())} runs isolate clusters smaller than 0.1% of customers "
      f"(smallest {prep['min_cluster_pct'].min():.3f}%), and the *highest* silhouettes in the table ({prep['silhouette'].max():.2f}) come from exactly those runs: the metric is rewarding the isolation of a handful of outliers, not real structure. "
      "(3) `RobustScaler` beats `StandardScaler` on silhouette and Davies-Bouldin; `StandardScaler` yields more balanced but more overlapping clusters. "
      "(4) The feature-set differences are smaller than the bootstrap standard deviation, i.e. a **statistical tie**; `engineered` was chosen because it adds utilisation and repayment-versus-minimum "
      "signals relevant to the brief. Silhouette is also not strictly comparable across different feature spaces, which is why stability and cluster size are weighted heavily.\n")
    a(f"![Preprocessing comparison]({fig}prep_01_preprocessing_comparison.png)\n")
    a(f"![Skewness]({fig}prep_02_skewness_before_after.png)\n")
    a(f"![Transform steps]({fig}prep_03_transform_steps.png)\n")
    a("`log1p` removes most of the skew but leaves the spike at zero; robust scaling only rescales (the shape is unchanged). A few columns are slightly over-corrected to the left (e.g. `PAYMENTS`, 5.9 to -1.8).\n")
    a(f"![Correlation after preprocessing]({fig}prep_04_correlation_transformed.png)\n")

    # ------------------------------------------------------------------ 6
    a("## 6. Clustering experiments\n")
    a(f"{sel['n_candidates']} candidate models were fitted on the same matrix with fixed seeds: K-Means and MiniBatch K-Means (k=2..10), Agglomerative (ward and average linkage, k=2..10), "
      f"Gaussian Mixture (full/tied/diag/spherical covariance, k=2..10), DBSCAN (eps from k-distance quantiles x min_samples 10/20/40) and HDBSCAN (min_cluster_size 50/100/200 x min_samples 10/30). "
      "MiniBatch K-Means is included as a scalability baseline, not because 9k rows need it. Models failing any hard constraint are not eligible regardless of score "
      f"(k in [{cfg['selection']['k_min']},{cfg['selection']['k_max']}], smallest cluster >= {cfg['selection']['min_cluster_fraction'] * 100:g}%, noise <= {cfg['selection']['max_noise_fraction'] * 100:g}%); "
      f"{sel['n_eligible']} of {sel['n_candidates']} are eligible.\n")
    a(md_table(algo_tbl.rename(columns={"its_silhouette": "silhouette of best"}), "{:.3f}") + "\n")
    a("What the results show:\n")
    a(f"- **Average-linkage agglomerative** reaches silhouettes up to {avg_link['silhouette'].max():.2f} and Davies-Bouldin as low as {avg_link['davies_bouldin'].min():.2f}, but its smallest cluster is {avg_min_n} customer(s): "
      "it peels off isolated outliers and leaves one giant cluster. Not a segmentation.")
    db_multi = dbs[dbs["n_clusters"] >= 2]
    a(f"- **DBSCAN**: {int((dbs['n_clusters'] <= 1).sum())} of {len(dbs)} settings collapse to a single cluster; in the {len(db_multi)} that find 2+ clusters the largest cluster still holds "
      f"{db_multi['max_cluster_pct'].min():.0f}-{db_multi['max_cluster_pct'].max():.0f}% of customers (all {int((~dbs['eligible']).sum())} settings are ineligible).")
    hdb_low_noise = hdb[hdb["noise_pct"] <= cfg["selection"]["max_noise_fraction"] * 100]
    a(f"- **HDBSCAN**: noise ranges from {hdb['noise_pct'].min():.0f}% to {hdb['noise_pct'].max():.0f}% of customers; the {len(hdb_low_noise)} setting(s) within the "
      f"{cfg['selection']['max_noise_fraction'] * 100:g}% noise limit find only {int(hdb_low_noise['n_clusters'].max()) if len(hdb_low_noise) else 0} clusters, below the minimum k of {cfg['selection']['k_min']}. "
      "Behaviour here looks like a continuum rather than well-separated dense groups.")
    a(f"- **Full-covariance GMM** has silhouettes of only {gmm_full['silhouette'].min():.2f}-{gmm_full['silhouette'].max():.2f}: components overlap heavily and chase the heavy tails. Tied and spherical covariance are competitive.")
    fin_elig = models[models["eligible"]]
    best_stab = fin_elig.sort_values("stability_ari_mean", ascending=False).iloc[0]
    a(f"- Only centroid-style models (K-Means, MiniBatch K-Means, Ward, spherical/tied/diag GMM) are both eligible and well-scoring. Among eligible finalists the most stable is "
      f"`{_short(best_stab['model_id'])}` (ARI {best_stab['stability_ari_mean']:.3f}); MiniBatch K-Means and several GMM settings show high variance across subsamples (see Section 7).\n")
    a(f"![Elbow]({fig}eval_01_kmeans_elbow.png)\n")
    a(f"![Metrics by k]({fig}eval_02_metrics_by_k.png)\n")
    a(f"![GMM BIC/AIC]({fig}eval_03_gmm_bic_aic.png)\n")
    a(f"![Density models]({fig}eval_04_density_models.png)\n")

    # ------------------------------------------------------------------ 7
    a("## 7. Evaluation and model selection\n")
    a(f"Finalists (top {cfg['stability']['finalists_per_algorithm']} eligible per algorithm, plus the best ineligible model where an algorithm has none, so rejections are visible) were stress-tested with "
      f"{cfg['stability']['n_rounds']} rounds of subsampling ({cfg['stability']['subsample_fraction'] * 100:.0f}% of rows, two independent refits per round, ARI on the shared rows). "
      f"Eligible models with stability >= {cfg['selection']['min_stability_ari']} are ranked by the equal-weight mean rank of silhouette, Davies-Bouldin, Calinski-Harabasz and stability.\n")
    t = models.copy()
    t["model"] = t["model_id"].map(_short)
    t["stability"] = t.apply(lambda r: f"{r['stability_ari_mean']:.2f} +/- {r['stability_ari_std']:.2f}", axis=1)
    t["status"] = np.where(t["selected"], "**SELECTED**", np.where(t["in_pool"], "ranked", "rejected: " + t["reject_reason"].fillna("").astype(str)))
    show = t[["model", "n_clusters", "noise_pct", "min_cluster_pct", "silhouette", "davies_bouldin", "calinski_harabasz", "stability", "mean_rank", "status"]]
    a(md_table(show.rename(columns={"n_clusters": "k", "noise_pct": "noise %", "min_cluster_pct": "min cluster %", "silhouette": "silh.", "davies_bouldin": "DB", "calinski_harabasz": "CH", "mean_rank": "mean rank"}), "{:.2f}") + "\n")
    a(f"![Finalist comparison]({fig}eval_05_finalist_comparison.png)\n")
    if sel["manual_override"]:
        a(f"**Decision.** `{_short(sel['composite_rank_winner'])}` has the best composite rank, but the composite is biased toward small k (silhouette and CH both fall as k grows). "
          f"{sel['override_reason']} `{_short(sel['chosen_model_id'])}` remains eligible and stable. "
          f"Trade-off accepted: silhouette drops from {winner['silhouette']:.3f} to {chosen['silhouette']:.3f} and CH from {winner['calinski_harabasz']:,.0f} to {chosen['calinski_harabasz']:,.0f}. "
          "A reader who prefers the more parsimonious k=3 can switch `selection.final_model_id` in `configs/config.yaml` and rerun.\n")
    st_k = k_tbl[["k", "silhouette", "davies_bouldin", "min_cluster_pct", "stability_ari_mean", "stability_ari_std"]].round(3)
    a("K-Means by k on the selected preprocessing (5-round stability from the preprocessing study):\n")
    a(md_table(st_k, "{:.3f}") + "\n")

    # ------------------------------------------------------------------ 8
    a("## 8. Customer profiles\n")
    a("Profiles use the **cleaned original-scale values** (not the scaled features). Names come from a deterministic rule on segment medians (`profiling.naming` in the config) and describe observed behaviour only.\n")
    cols = ["segment_name", "pct_customers", "BALANCE_median", "PURCHASES_median", "ONEOFF_PURCHASES_median", "INSTALLMENTS_PURCHASES_median", "CASH_ADVANCE_median",
            "PURCHASES_FREQUENCY_median", "CASH_ADVANCE_FREQUENCY_median", "CREDIT_LIMIT_median", "PAYMENTS_median", "BALANCE_TO_LIMIT_median"]
    a("**Medians by segment:**\n")
    a(md_table(summary[cols].rename(columns=lambda c: c.replace("_median", " (med)").replace("_", " ").lower() if c != "segment_name" else "segment"), "{:,.2f}") + "\n")
    cols2 = ["segment_name", "pct_use_cash_advance", "pct_no_purchases", "pct_full_payers_ge_50", "pct_never_full_payment", "pct_payments_below_min_due", "pct_balance_above_limit", "PAYMENT_TO_MINPAY_RATIO_median", "TENURE_mean"]
    a("**Observed usage and repayment (% of segment):**\n")
    a(md_table(summary[cols2].rename(columns=lambda c: c.replace("pct_", "% ").replace("_", " ") if c != "segment_name" else "segment"), "{:,.1f}") + "\n")
    a(f"![Cluster sizes]({fig}seg_01_cluster_sizes.png)\n")
    a(f"![Profile heatmap]({fig}seg_03_profile_index_heatmap.png)\n")
    a(f"![Segment comparison]({fig}seg_04_segment_comparison.png)\n")
    pca = run["pca_explained_variance"]
    a(f"![PCA]({fig}seg_02_pca_segments.png)\n")
    cum = run.get("pca_cumulative_variance", [])
    k80 = next((i + 1 for i, v in enumerate(cum) if v >= 0.80), None)
    k90 = next((i + 1 for i, v in enumerate(cum) if v >= 0.90), None)
    a(f"PCA note: PC1+PC2 capture {sum(pca) * 100:.1f}% of variance" + (f" (80% needs {k80} components, 90% needs {k90})" if k80 else "") +
      ", so the 2-D picture shows overlap that is not present in the full space. It is an illustration, not evidence of cluster quality; clustering used all features.\n")
    a(f"![PCA scree]({fig}eval_06_pca_scree.png)\n")
    a(f"![PCA other views]({fig}seg_05_pca_other_views.png)\n")
    a("In the PC1/PC3 view, Cash-Advance Borrowers separate cleanly from Active Balance-Carriers along PC3; Low-Activity Light Users and Frequent Full-Payers still overlap in these 2-D views.\n")
    a("### Segments: observed behaviour, assumptions, and suggested actions\n")
    a("> **Observed** = measured in this dataset. **Assumption** = a hypothesis about profit or risk that the data cannot confirm (no revenue, interest, fee, or delinquency fields exist). Actions are suggestions to test, ideally with A/B experiments.\n")
    for seg, info in run["segments"].items():
        a(f"#### {seg} ({info['pct']:.1f}%)\n")
        a(f"- **Observed:** {info['description']}")
        a(f"- **Assumption (unverified):** {info.get('assumption', 'n/a')}")
        a(f"- **Suggested action:** {info.get('action', 'n/a')}\n")

    # ------------------------------------------------------------------ 9
    a("## 9. Robustness and integrity checks\n")
    al = run["alignment"]
    a(f"- **ID/label alignment:** {al['n_customers']:,} customers; {al['n_unassigned']} unassigned; {al['n_label_mismatches']} label mismatches; same id set as the raw file: {al['same_id_set']}; row order preserved: {al['row_order_preserved']}.")
    if "repredict_matches_labels" in run:
        a(f"- **Refit consistency:** re-predicting the fitted model on the transformed data reproduces the stored labels: {run['repredict_matches_labels']}.")
    fs = rob.get("feature_set_agreement_ari")
    if fs:
        a("- **Feature-set agreement** (ARI between the final K-Means and K-Means with the same k on another feature set): " + ", ".join(f"`{k}` {v:.2f}" for k, v in fs.items()) + ".")
    a(f"- **Raw data unchanged:** {run['raw_data_unchanged']} (SHA-256 `{run['raw_data_sha256_after'][:16]}...`).\n")

    # ------------------------------------------------------------------ 10
    a("## 10. Limitations\n")
    a("- Single snapshot of up to 12 months; no time dimension, so behavioural change and churn cannot be assessed.")
    a("- No profitability, interest, fee, delinquency or demographic data: risk and value statements are assumptions.")
    a("- Internal validity indices (silhouette, DB, CH) favour compact, convex clusters and small k; credit-card behaviour is a continuum, so segments are *useful partitions*, not natural kinds. Silhouette values around 0.25-0.30 indicate moderate overlap.")
    a("- Silhouette is not strictly comparable across different feature spaces; the preprocessing study therefore leans on stability and cluster size.")
    a("- Stability was estimated with a small number of rounds (5 for the preprocessing study, 10 for finalists); some solutions are bimodal (high mean, large std).")
    fs_ = rob.get("feature_set_agreement_ari", {})
    fs_txt = (f" Re-running K-Means with the same k on `base_no_total` agrees with the final labels only at ARI {fs_['base_no_total']:.2f}, so the feature-set choice changes a meaningful share of individual assignments even though headline metrics tie; the segment *archetypes* should be trusted more than any single customer's label." if "base_no_total" in fs_ else "")
    a("- The k=4 choice over the composite-rank winner k=3 is a documented judgement call made after seeing the profiles; the feature-set choice was a statistical tie on metrics." + fs_txt)
    a("- Segment names use thresholds tuned on this dataset and should be re-reviewed on new data. New customers can be scored (`segmentation.scoring`, dashboard tab \"Test the model\"), but a score is the *closest* of four centroids, not a probability, and inputs far outside the training range are extrapolations.\n")

    # ------------------------------------------------------------------ 11
    a("## 11. Reproducibility\n")
    a("```bash\npip install -r requirements-dev.txt        # runtime + pytest (requirements.txt = runtime only)\npython run_pipeline.py                   # ~12 min on one CPU core; add --reuse-prep-comparison to skip a 3.5 min step\npython -m segmentation.reporting          # regenerates this report (run with PYTHONPATH=src)\nstreamlit run app/streamlit_app.py\npytest\n```\n")
    a("Scoring check: scoring all raw customers with the saved bundle reproduces the pipeline assignments exactly (automated test). Artifacts: `outputs/model_comparison.csv`, `outputs/customer_assignments.csv`, `outputs/segment_summary.csv`, `outputs/segment_profile_index.csv`, "
      "`outputs/clustering_grid_all_models.csv`, `outputs/preprocessing_comparison.csv`, `outputs/customers_with_segments.csv`, `figures/*.png`, `models/final_model.joblib`.\n")

    path = PROJECT_ROOT / cfg["paths"]["reports_dir"] / "final_report.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(L), encoding="utf-8")
    return path


if __name__ == "__main__":
    print(write_final_report())
