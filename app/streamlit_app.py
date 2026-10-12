"""Streamlit dashboard: explore segments, EDA, preprocessing, and score new customers.

Run the pipeline first (``python run_pipeline.py``), then::

    streamlit run app/streamlit_app.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from segmentation.scoring import load_bundle, score_customers  # noqa: E402

OUT = ROOT / "outputs"
FIG = ROOT / "figures"
MODEL_PATH = ROOT / "models" / "final_model.joblib"
ID_COL = "CUST_ID"
MAX_BATCH_ROWS = 20_000

st.set_page_config(page_title="Credit-Card Customer Segments", layout="wide")


# ------------------------------------------------------------------ loading
@st.cache_data(show_spinner=False)
def load_outputs() -> dict:
    def read(name: str):
        p = OUT / name
        return pd.read_csv(p) if p.exists() else None

    rs = OUT / "run_summary.json"
    return {
        "customers": read("customers_with_segments.csv"),
        "summary": read("segment_summary.csv"),
        "index": read("segment_profile_index.csv"),
        "models": read("model_comparison.csv"),
        "audit": read("data_audit_columns.csv"),
        "corr_pairs": read("data_audit_top_correlations_spearman.csv"),
        "prep": read("preprocessing_comparison.csv"),
        "high_corr": read("preprocessing_high_correlation_pairs.csv"),
        "run": json.loads(rs.read_text()) if rs.exists() else None,
    }


@st.cache_resource(show_spinner=False)
def get_bundle():
    return load_bundle(MODEL_PATH)


def to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


def show_figure(name: str, title: str, takeaway: str) -> None:
    path = FIG / name
    st.markdown(f"**{title}**")
    if path.exists():
        st.image(str(path), width="stretch")
    else:
        st.warning(f"Figure {name} not found - re-run the pipeline.")
    st.caption(takeaway)


data = load_outputs()
if data["customers"] is None or data["summary"] is None or data["run"] is None:
    st.error("Pipeline outputs not found. Run `python run_pipeline.py` from the project root first, then reload this page.")
    st.stop()

cust, summary, index_df, models, run = data["customers"], data["summary"], data["index"], data["models"], data["run"]
names = summary.sort_values("cluster")["segment_name"].tolist()

st.title("Credit-Card Customer Segmentation")
sel = run["selection"]
st.caption(f"Selected model: `{sel['chosen_model_id']}` | {len(cust):,} customers | seed {run['seed']} | "
           f"PCA(2) explains {sum(run['pca_explained_variance']) * 100:.1f}% of variance (all 19 features were used for clustering)")

# ------------------------------------------------------------------ sidebar filters
st.sidebar.header("Filters")
st.sidebar.caption("Filters apply to the Overview, Feature explorer and Downloads tabs.")
chosen = st.sidebar.multiselect("Segments", names, default=names)
lo, hi = float(cust["CREDIT_LIMIT"].min()), float(cust["CREDIT_LIMIT"].max())
limit_range = st.sidebar.slider("Credit limit", lo, hi, (lo, hi))
bal_hi = float(cust["BALANCE"].max())
bal_range = st.sidebar.slider("Balance", 0.0, bal_hi, (0.0, bal_hi))
view = cust[cust["segment_name"].isin(chosen) & cust["CREDIT_LIMIT"].between(*limit_range) & cust["BALANCE"].between(*bal_range)]
st.sidebar.metric("Customers matching filters", f"{len(view):,}")

(tab_over, tab_prof, tab_eda, tab_prep, tab_test, tab_feat, tab_cust, tab_model, tab_dl) = st.tabs(
    ["Overview", "Segment profiles", "EDA", "Preprocessing", "Test the model", "Feature explorer", "Customer lookup", "Model comparison", "Downloads"]
)

# ------------------------------------------------------------------ overview
with tab_over:
    c1, c2 = st.columns([1, 1.4])
    with c1:
        st.subheader("Segment sizes")
        sizes = view["segment_name"].value_counts().reindex(names).dropna().astype(int)
        st.bar_chart(sizes)
        st.dataframe(summary[["cluster", "segment_name", "n_customers", "pct_customers"]], hide_index=True, width="stretch")
    with c2:
        st.subheader("PCA projection (PC1 vs PC2)")
        sample = view.sample(min(len(view), 4000), random_state=0) if len(view) else view
        st.scatter_chart(sample, x="pca_1", y="pca_2", color="segment_name", size=8)
        st.caption("PC1+PC2 capture about 61% of the variance, so segments overlap here more than they do in the full feature space.")
    c3, c4 = st.columns(2)
    with c3:
        show_figure("eval_06_pca_scree.png", "How much variance does PCA capture?", "Four components are needed for 80% and seven for 90%, so no 2-D plot tells the whole story.")
    with c4:
        show_figure("seg_05_pca_other_views.png", "Other PCA views (PC3)", "PC3 cleanly separates Cash-Advance Borrowers from Active Balance-Carriers; Low-Activity Light Users and Frequent Full-Payers still overlap in these 2-D views.")

# ------------------------------------------------------------------ profiles
with tab_prof:
    st.subheader("Observed behaviour by segment (medians)")
    cols = ["segment_name", "pct_customers", "BALANCE_median", "PURCHASES_median", "CASH_ADVANCE_median", "CREDIT_LIMIT_median",
            "PAYMENTS_median", "PURCHASES_FREQUENCY_median", "CASH_ADVANCE_FREQUENCY_median", "BALANCE_TO_LIMIT_median",
            "pct_full_payers_ge_50", "pct_payments_below_min_due"]
    st.dataframe(summary[cols].round(2), hide_index=True, width="stretch")
    if index_df is not None:
        st.subheader("Profile index (segment mean / population mean x 100)")
        idx = index_df.set_index(index_df.columns[0])
        show = [c for c in ["BALANCE", "PURCHASES", "ONEOFF_PURCHASES", "INSTALLMENTS_PURCHASES", "CASH_ADVANCE", "PURCHASES_FREQUENCY",
                            "CASH_ADVANCE_FREQUENCY", "CREDIT_LIMIT", "PAYMENTS", "PRC_FULL_PAYMENT", "BALANCE_TO_LIMIT"] if c in idx.columns]
        fig, ax = plt.subplots(figsize=(11, 0.7 * len(idx) + 2))
        sns.heatmap(idx[show] / 100, annot=idx[show].round(0).astype(int), fmt="d", cmap="RdBu_r", center=1, vmin=0, vmax=3, linewidths=0.4, ax=ax)
        ax.set_ylabel("")
        st.pyplot(fig)
        plt.close(fig)
    show_figure("seg_04_segment_comparison.png", "Segment comparison", "Medians on observed behaviour; no profitability or risk data exists in this dataset.")
    st.subheader("Observed behaviour vs. assumptions")
    st.info("The dataset has no revenue, interest, fee or delinquency fields. Profitability and risk statements below are **assumptions to test**, not findings.")
    for seg, info in run["segments"].items():
        with st.expander(f"{seg} - {info['pct']:.1f}% of customers"):
            st.markdown(f"**Observed:** {info['description']}")
            st.markdown(f"**Assumption (unverified):** {info.get('assumption', 'n/a')}")
            st.markdown(f"**Suggested action:** {info.get('action', 'n/a')}")

# ------------------------------------------------------------------ EDA
with tab_eda:
    st.subheader("Exploratory data analysis")
    cons = run["consistency"]
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Customers", f"{cons['n_rows']:,}")
    m2.metric("Missing MINIMUM_PAYMENTS", f"{cons['min_payments_missing']}")
    m3.metric("PURCHASES = ONEOFF + INSTALLMENTS", f"{cons['purchases_equal_components_pct']:.2f}%")
    m4.metric("Balance above credit limit", f"{cons['balance_above_credit_limit']}")
    pairs = [
        ("eda_01_missing_values.png", "Missing values", "MINIMUM_PAYMENTS has 313 missing values (3.5%); CREDIT_LIMIT has 1."),
        ("eda_02_min_payments_missing_mechanism.png", "Why MINIMUM_PAYMENTS is missing", "Every customer with zero payments lacks it (240), plus 73 customers who did pay: not missing at random, so a flag was added before imputing."),
        ("eda_03_distributions_raw.png", "Raw distributions", "Amount features are extremely right-skewed with long tails; frequencies are bounded in [0, 1]."),
        ("eda_04_distributions_log1p.png", "Distributions after log1p", "log1p makes the amount features far more symmetric; the spike at zero (customers who never use a feature) remains and is real behaviour."),
        ("eda_05_correlation_spearman.png", "Spearman correlation (raw data)", "Purchase-related fields move together; strongly correlated pairs are listed below."),
        ("eda_06_outlier_boxplots.png", "Outliers (log scale)", "Long upper tails of legitimate high-value customers. No rows were removed."),
        ("eda_09_outlier_share.png", "Share of extreme customers per feature", "Percentage above Q3 + 3 x IQR: tails are real but small shares of the base."),
        ("eda_07_purchase_redundancy.png", "PURCHASES is almost the sum of its parts", "PURCHASES was dropped from the model to avoid double-counting purchase behaviour."),
        ("eda_08_correlation_model_features.png", "Correlation among the model's input features", "Used to check redundancy; highly correlated pairs after transformation are on the Preprocessing tab."),
    ]
    for i in range(0, len(pairs), 2):
        cols_ = st.columns(2)
        for col, p in zip(cols_, pairs[i:i + 2]):
            with col:
                show_figure(*p)
    if data["corr_pairs"] is not None:
        with st.expander("Strongly correlated raw feature pairs (|Spearman| >= 0.6)"):
            st.dataframe(data["corr_pairs"].round(3), hide_index=True, width="stretch")
    if data["audit"] is not None:
        with st.expander("Per-column audit table (missing, zeros, skew, outliers)"):
            st.dataframe(data["audit"].round(3), hide_index=True, width="stretch")

# ------------------------------------------------------------------ preprocessing
with tab_prep:
    st.subheader("Preprocessing pipeline")
    pp = run["preprocessing"]
    st.markdown(
        f"`SimpleImputer(median)` safety net -> `log1p` on **{len(pp['log_columns'])}** skewed, unbounded, non-negative columns -> "
        f"`{pp['scaler'].title()}Scaler` -> `VarianceThreshold` (dropped constant features: {pp['dropped_by_variance'] or 'none'}). "
        f"Input: {pp['n_features_in']} features; output: {pp['n_features_out']}."
    )
    st.markdown("**log1p columns:** " + ", ".join(f"`{c}`" for c in pp["log_columns"]))
    c1, c2 = st.columns(2)
    with c1:
        show_figure("prep_03_transform_steps.png", "Raw -> log1p -> robust scaling", "log1p removes most of the skew (e.g. MINIMUM_PAYMENTS 13.9 -> 0.3) but leaves the spike at zero; a few columns are slightly over-corrected (PAYMENTS 5.9 -> -1.8). Robust scaling only rescales (shape unchanged). Proportions in [0, 1] are left as-is.")
    with c2:
        show_figure("prep_02_skewness_before_after.png", "Skewness before/after", "Absolute skew falls for all 12 log-transformed columns.")
    c3, c4 = st.columns(2)
    with c3:
        show_figure("prep_04_correlation_transformed.png", "Correlation in the model input space", "A few pairs exceed 0.9 (e.g. cash-advance count vs frequency); they were reported, and removing the two *_TRX columns did not improve stability.")
    with c4:
        show_figure("prep_01_preprocessing_comparison.png", "16 preprocessing variants compared", "log1p variants are most stable; very high silhouettes in raw variants come from clusters of ~2 customers.")
    if data["prep"] is not None:
        p = data["prep"].copy()
        p["variant"] = p["feature_set"] + " | " + np.where(p["log1p"], "log1p", "raw") + " + " + p["scaler"]
        tbl = (p.groupby("variant").agg(silhouette=("silhouette", "mean"), davies_bouldin=("davies_bouldin", "mean"), stability_ari=("stability_ari_mean", "mean"),
                                        worst_min_cluster_pct=("min_cluster_pct", "min")).sort_values("stability_ari", ascending=False).round(3).reset_index())
        st.markdown("**Variant comparison (K-Means, mean over k = 3..8)**")
        st.dataframe(tbl, hide_index=True, width="stretch")
    sens = run.get("robustness", {}).get("imputation_sensitivity")
    if sens:
        st.markdown("**MINIMUM_PAYMENTS imputation sensitivity** (ARI vs. the median strategy; 1 = identical segmentation)")
        st.dataframe(pd.DataFrame(sens).round(4), hide_index=True, width="stretch")
    if data["high_corr"] is not None and len(data["high_corr"]):
        with st.expander("Feature pairs with |r| >= 0.9 after preprocessing"):
            st.dataframe(data["high_corr"].round(3), hide_index=True, width="stretch")

# ------------------------------------------------------------------ test the model
with tab_test:
    st.subheader("Test the model on a new customer")
    st.caption("Enter raw customer features exactly as in the original data. The app cleans them like the training data, applies the saved "
               "preprocessing and K-Means model, and returns the closest segment. This describes behaviour; it is not a credit decision.")
    try:
        bundle = get_bundle()
    except (FileNotFoundError, KeyError) as exc:
        st.error(str(exc))
        st.stop()

    seg_idx = summary.set_index("segment_name")
    mode = st.radio("Mode", ["Single customer", "Batch (CSV upload)"], horizontal=True)
    amount_cols = ["BALANCE", "CREDIT_LIMIT", "PAYMENTS", "MINIMUM_PAYMENTS", "ONEOFF_PURCHASES", "INSTALLMENTS_PURCHASES", "CASH_ADVANCE"]
    count_cols = ["PURCHASES_TRX", "CASH_ADVANCE_TRX"]
    prop_cols = ["BALANCE_FREQUENCY", "PURCHASES_FREQUENCY", "ONEOFF_PURCHASES_FREQUENCY", "PURCHASES_INSTALLMENTS_FREQUENCY", "CASH_ADVANCE_FREQUENCY", "PRC_FULL_PAYMENT"]

    if mode == "Single customer":
        preset = st.selectbox("Start from the typical (median) customer of:", names + ["Blank customer (all zeros)"])
        def default(col: str) -> float:
            if preset.startswith("Blank"):
                return 0.0
            return float(seg_idx.loc[preset, f"{col}_median"])

        with st.form("score_form"):
            a, b, c = st.columns(3)
            vals: dict[str, float] = {}
            with a:
                st.markdown("**Balance, limit, repayment**")
                for col in ["BALANCE", "CREDIT_LIMIT", "PAYMENTS"]:
                    vals[col] = st.number_input(col, min_value=0.0, value=round(default(col), 2), step=100.0, key=f"{preset}_{col}")
                min_unknown = st.checkbox("MINIMUM_PAYMENTS unknown (use training rule)", value=False, key=f"{preset}_minunk")
                vals["MINIMUM_PAYMENTS"] = st.number_input("MINIMUM_PAYMENTS", min_value=0.0, value=round(default("MINIMUM_PAYMENTS"), 2), step=50.0, key=f"{preset}_MINIMUM_PAYMENTS")
                vals["PRC_FULL_PAYMENT"] = st.number_input("PRC_FULL_PAYMENT (0-1)", min_value=0.0, max_value=1.0, value=round(default("PRC_FULL_PAYMENT"), 3), step=0.05, key=f"{preset}_PRC")
                vals["BALANCE_FREQUENCY"] = st.number_input("BALANCE_FREQUENCY (0-1)", min_value=0.0, max_value=1.0, value=round(default("BALANCE_FREQUENCY"), 3), step=0.05, key=f"{preset}_BF")
            with b:
                st.markdown("**Purchases**")
                for col in ["ONEOFF_PURCHASES", "INSTALLMENTS_PURCHASES"]:
                    vals[col] = st.number_input(col, min_value=0.0, value=round(default(col), 2), step=100.0, key=f"{preset}_{col}")
                vals["PURCHASES_TRX"] = float(st.number_input("PURCHASES_TRX", min_value=0, value=int(round(default("PURCHASES_TRX"))), step=1, key=f"{preset}_PTRX"))
                for col in ["PURCHASES_FREQUENCY", "ONEOFF_PURCHASES_FREQUENCY", "PURCHASES_INSTALLMENTS_FREQUENCY"]:
                    vals[col] = st.number_input(f"{col} (0-1)", min_value=0.0, max_value=1.0, value=round(default(col), 3), step=0.05, key=f"{preset}_{col}")
            with c:
                st.markdown("**Cash advance**")
                vals["CASH_ADVANCE"] = st.number_input("CASH_ADVANCE", min_value=0.0, value=round(default("CASH_ADVANCE"), 2), step=100.0, key=f"{preset}_CA")
                vals["CASH_ADVANCE_TRX"] = float(st.number_input("CASH_ADVANCE_TRX", min_value=0, value=int(round(default("CASH_ADVANCE_TRX"))), step=1, key=f"{preset}_CATRX"))
                vals["CASH_ADVANCE_FREQUENCY"] = st.number_input("CASH_ADVANCE_FREQUENCY (0-1)", min_value=0.0, max_value=1.0, value=round(default("CASH_ADVANCE_FREQUENCY"), 3), step=0.05, key=f"{preset}_CAF")
            submitted = st.form_submit_button("Assign segment", type="primary")

        if submitted:
            row = dict(vals)
            if min_unknown:
                row["MINIMUM_PAYMENTS"] = np.nan
            if row["CREDIT_LIMIT"] <= 0:
                st.error("CREDIT_LIMIT must be greater than 0.")
            else:
                try:
                    res, notes = score_customers(pd.DataFrame([row]), bundle)
                except ValueError as exc:
                    st.error(str(exc))
                else:
                    r = res.iloc[0]
                    st.success(f"Assigned segment: **{r['segment_name']}** (cluster {int(r['cluster'])})")
                    if "margin" in res:
                        mcol1, mcol2 = st.columns([1, 2])
                        mcol1.metric("Assignment margin", f"{r['margin']:.2f}", help="1 - nearest/second-nearest centroid distance. Near 0 = borderline between two segments; this is a closeness heuristic, not a probability.")
                        if r["margin"] < 0.15:
                            mcol1.warning("Borderline: close to another segment as well.")
                        dist = pd.Series({c.replace("dist_", ""): r[c] for c in res.columns if c.startswith("dist_")}, name="distance")
                        mcol2.markdown("**Distance to each segment centroid** (lower = closer)")
                        mcol2.bar_chart(dist)
                    for n in notes:
                        st.warning(n)
                    seg = r["segment_name"]
                    cmp_feats = ["BALANCE", "PURCHASES", "CASH_ADVANCE", "CREDIT_LIMIT", "PAYMENTS", "PURCHASES_FREQUENCY", "CASH_ADVANCE_FREQUENCY", "PRC_FULL_PAYMENT"]
                    entered = dict(row)
                    entered["PURCHASES"] = row["ONEOFF_PURCHASES"] + row["INSTALLMENTS_PURCHASES"]
                    st.markdown("**Entered values vs. the median customer of the assigned segment**")
                    st.dataframe(pd.DataFrame({"entered": [entered[f] for f in cmp_feats], f"{seg} median": [seg_idx.loc[seg, f"{f}_median"] for f in cmp_feats]}, index=cmp_feats).round(3), width="stretch")
                    info = run["segments"].get(seg, {})
                    if info:
                        with st.expander("What this segment means (observed vs. assumed)", expanded=True):
                            st.markdown(f"**Observed:** {info['description']}")
                            st.markdown(f"**Assumption (unverified):** {info.get('assumption', 'n/a')}")
                            st.markdown(f"**Suggested action:** {info.get('action', 'n/a')}")
    else:
        required = [c for c in bundle["input_columns"]]
        template = pd.DataFrame([{c: float(seg_idx.iloc[0][f"{c}_median"]) for c in required}])
        st.download_button("Download CSV template (one example row)", to_csv_bytes(template), "scoring_template.csv", "text/csv")
        st.caption(f"Required columns: {', '.join(required)}. CREDIT_LIMIT and MINIMUM_PAYMENTS may be blank. Optional: {ID_COL}. Max {MAX_BATCH_ROWS:,} rows.")
        up = st.file_uploader("Upload a CSV of customers", type=["csv"])
        if up is not None:
            try:
                df_up = pd.read_csv(up)
            except Exception as exc:  # noqa: BLE001 - show any parse problem to the user
                st.error(f"Could not read the CSV: {exc}")
                df_up = None
            if df_up is not None:
                if len(df_up) > MAX_BATCH_ROWS:
                    st.error(f"Too many rows ({len(df_up):,}); the limit is {MAX_BATCH_ROWS:,}.")
                elif df_up.empty:
                    st.error("The file has no rows.")
                else:
                    try:
                        res, notes = score_customers(df_up, bundle)
                    except ValueError as exc:
                        st.error(str(exc))
                    else:
                        out_df = pd.concat([df_up, res], axis=1)
                        st.success(f"Scored {len(out_df):,} customers.")
                        for n in notes:
                            st.warning(n)
                        st.bar_chart(res["segment_name"].value_counts())
                        st.dataframe(out_df.head(200), width="stretch")
                        st.download_button("Download scored customers (CSV)", to_csv_bytes(out_df), "scored_customers.csv", "text/csv")

# ------------------------------------------------------------------ feature explorer
with tab_feat:
    numeric = [c for c in cust.columns if c not in (ID_COL, "cluster", "pca_1", "pca_2") and pd.api.types.is_numeric_dtype(cust[c])
               and not c.endswith("_MISSING") and c != "PURCHASE_COMPONENT_MISMATCH"]
    feat = st.selectbox("Feature", numeric, index=numeric.index("PURCHASES") if "PURCHASES" in numeric else 0)
    log_scale = st.checkbox("log1p scale", value=feat in {"BALANCE", "PURCHASES", "CASH_ADVANCE", "PAYMENTS", "CREDIT_LIMIT"})
    if len(view):
        fig, ax = plt.subplots(figsize=(10, 4.5))
        plot_df = view.assign(_v=np.log1p(view[feat].clip(lower=0)) if log_scale else view[feat])
        sns.boxplot(data=plot_df, x="segment_name", y="_v", order=[n for n in names if n in chosen], ax=ax)
        ax.set_xlabel("")
        ax.set_ylabel(f"log1p({feat})" if log_scale else feat)
        plt.setp(ax.get_xticklabels(), rotation=15, ha="right")
        st.pyplot(fig)
        plt.close(fig)
        st.dataframe(view.groupby("segment_name")[feat].describe().round(2), width="stretch")
    else:
        st.warning("No customers match the current filters.")

# ------------------------------------------------------------------ customer lookup
with tab_cust:
    cid = st.text_input("Customer ID", value=cust[ID_COL].iloc[0]).strip()
    row = cust[cust[ID_COL].str.upper() == cid.upper()]
    if row.empty:
        st.warning(f"No customer with id '{cid}'.")
    else:
        r = row.iloc[0]
        st.success(f"{r[ID_COL]} belongs to **{r['segment_name']}** (cluster {int(r['cluster'])})")
        seg_med = summary.set_index("cluster").loc[int(r["cluster"])]
        feats = ["BALANCE", "PURCHASES", "CASH_ADVANCE", "CREDIT_LIMIT", "PAYMENTS", "PURCHASES_FREQUENCY", "CASH_ADVANCE_FREQUENCY", "PRC_FULL_PAYMENT", "BALANCE_TO_LIMIT"]
        st.dataframe(pd.DataFrame({"customer": [r[f] for f in feats], "segment median": [seg_med[f"{f}_median"] for f in feats]}, index=feats).round(3))

# ------------------------------------------------------------------ model comparison
with tab_model:
    if models is not None:
        st.subheader("Finalist models")
        show = models[["model_id", "n_clusters", "noise_pct", "min_cluster_pct", "silhouette", "davies_bouldin", "calinski_harabasz",
                       "stability_ari_mean", "eligible", "stable", "mean_rank", "selected", "reject_reason"]].round(3)
        st.dataframe(show, hide_index=True, width="stretch")
        if sel["manual_override"]:
            st.warning(f"The chosen model differs from the composite-rank winner (`{sel['composite_rank_winner']}`). Reason: {sel['override_reason']}")
    for f, t in [("eval_01_kmeans_elbow.png", "K-Means elbow"), ("eval_02_metrics_by_k.png", "Metrics vs. k"), ("eval_05_finalist_comparison.png", "Finalists")]:
        if (FIG / f).exists():
            st.markdown(f"**{t}**")
            st.image(str(FIG / f), width="stretch")

# ------------------------------------------------------------------ downloads
with tab_dl:
    st.write(f"Filtered customers: **{len(view):,}**")
    st.download_button("Download filtered customers (CSV)", to_csv_bytes(view), "filtered_customers.csv", "text/csv")
    st.download_button("Download all assignments (CSV)", to_csv_bytes(cust[[ID_COL, "cluster", "segment_name"]]), "customer_assignments.csv", "text/csv")
    st.download_button("Download segment summary (CSV)", to_csv_bytes(summary), "segment_summary.csv", "text/csv")
    if models is not None:
        st.download_button("Download model comparison (CSV)", to_csv_bytes(models), "model_comparison.csv", "text/csv")
