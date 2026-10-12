"""All figures. Every function saves a PNG and returns its path."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sns.set_theme(style="whitegrid", context="notebook")
PALETTE = sns.color_palette("tab10")


def _save(fig: plt.Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return path


# --------------------------------------------------------------------- EDA
def plot_missing(df: pd.DataFrame, out: Path) -> Path:
    miss = df.isna().sum()
    miss = miss[miss > 0].sort_values()
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.barh(miss.index, miss.values, color=PALETTE[3])
    for y, v in enumerate(miss.values):
        ax.text(v, y, f" {v} ({v / len(df) * 100:.2f}%)", va="center")
    ax.set_xlabel("Missing values")
    ax.set_title("Missing values by column (raw data)")
    ax.set_xlim(0, miss.max() * 1.35)
    return _save(fig, out)


def plot_distributions(df: pd.DataFrame, cols: list[str], out: Path, log1p: bool = False) -> Path:
    n = len(cols)
    ncol = 4
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(16, 2.8 * nrow))
    for ax, c in zip(axes.ravel(), cols):
        s = df[c].dropna()
        s = np.log1p(s) if log1p else s
        ax.hist(s, bins=50, color=PALETTE[0], alpha=0.85)
        ax.set_title(f"{c}\nskew={s.skew():.2f}", fontsize=9)
        ax.tick_params(labelsize=7)
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    fig.suptitle("Feature distributions " + ("(log1p)" if log1p else "(raw)"), y=1.01, fontsize=13)
    return _save(fig, out)


def plot_skewness(df: pd.DataFrame, cols: list[str], out: Path, log_cols: list[str] | None = None) -> Path:
    raw = df[cols].skew()
    order = raw.sort_values().index
    fig, ax = plt.subplots(figsize=(8, 6))
    y = np.arange(len(order))
    ax.barh(y - 0.2, raw[order], height=0.4, label="raw", color=PALETTE[3])
    if log_cols is not None:
        after = pd.Series({c: (np.log1p(df[c]).skew() if c in log_cols else df[c].skew()) for c in cols})
        ax.barh(y + 0.2, after[order], height=0.4, label="after log1p (eligible columns)", color=PALETTE[2])
        ax.legend()
    ax.set_yticks(y)
    ax.set_yticklabels(order, fontsize=8)
    ax.axvline(0, color="k", lw=0.8)
    ax.set_xlabel("Skewness")
    ax.set_title("Skewness before/after transformation")
    return _save(fig, out)


def plot_correlation(df: pd.DataFrame, cols: list[str], out: Path, method: str = "spearman", title: str | None = None) -> Path:
    corr = df[cols].corr(method=method)
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    fig, ax = plt.subplots(figsize=(11, 9))
    sns.heatmap(corr, mask=mask, cmap="coolwarm", vmin=-1, vmax=1, center=0, annot=True, fmt=".2f", annot_kws={"size": 6}, linewidths=0.3, ax=ax)
    ax.set_title(title or f"{method.title()} correlation")
    return _save(fig, out)


def plot_outlier_boxplots(df: pd.DataFrame, cols: list[str], out: Path) -> Path:
    fig, ax = plt.subplots(figsize=(12, 5))
    data = [np.log1p(df[c].clip(lower=0)) for c in cols]
    ax.boxplot(data, tick_labels=cols, vert=True, flierprops={"marker": ".", "markersize": 2, "alpha": 0.4})
    ax.set_ylabel("log1p(value)")
    ax.set_title("Outliers per feature (log1p scale for visibility; no rows removed)")
    plt.setp(ax.get_xticklabels(), rotation=60, ha="right", fontsize=8)
    return _save(fig, out)


def plot_purchase_redundancy(df: pd.DataFrame, out: Path) -> Path:
    comp = df["ONEOFF_PURCHASES"] + df["INSTALLMENTS_PURCHASES"]
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.scatter(comp, df["PURCHASES"], s=4, alpha=0.4, color=PALETTE[0])
    lim = max(comp.max(), df["PURCHASES"].max())
    ax.plot([0, lim], [0, lim], "r--", lw=1, label="y = x")
    ax.set_xlabel("ONEOFF + INSTALLMENTS purchases")
    ax.set_ylabel("PURCHASES")
    ax.set_title("PURCHASES is (almost) the sum of its components")
    ax.legend()
    return _save(fig, out)


def plot_missing_mechanism(df: pd.DataFrame, out: Path) -> Path:
    """MINIMUM_PAYMENTS missingness vs. whether any payment was made (grouped bars, counts labelled)."""
    miss = df["MINIMUM_PAYMENTS"].isna().map({True: "MIN_PAYMENTS missing", False: "MIN_PAYMENTS present"})
    pay = df["PAYMENTS"].eq(0).map({True: "PAYMENTS == 0", False: "PAYMENTS > 0"})
    tab = pd.crosstab(pay, miss)
    fig, ax = plt.subplots(figsize=(7, 4))
    tab.plot(kind="bar", ax=ax, color=[PALETTE[3], PALETTE[0]], rot=0, width=0.7)
    ax.set_yscale("log")
    ax.set_ylim(1, tab.to_numpy().max() * 5)
    for cont in ax.containers:
        ax.bar_label(cont, fmt="%d", padding=2)
    ax.set_xlabel("")
    ax.set_ylabel("Customers (log scale)")
    ax.set_title("MINIMUM_PAYMENTS is missing for every zero-payment customer\n(plus 73 customers who did pay)")
    ax.legend(title="")
    return _save(fig, out)


# ------------------------------------------------------ preprocessing / evaluation
def plot_preprocessing_comparison(prep: pd.DataFrame, out: Path) -> Path:
    """Left: stability per variant. Right: silhouette vs smallest-cluster size (exposes inflated scores)."""
    d = prep.copy()
    d["variant"] = d["feature_set"] + " | " + np.where(d["log1p"], "log1p", "raw") + " + " + d["scaler"]
    agg = d.groupby("variant").agg(stab=("stability_ari_mean", "mean")).sort_values("stab")
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.5), gridspec_kw={"width_ratios": [1.1, 1]})
    colors = [PALETTE[2] if "log1p" in v else PALETTE[3] for v in agg.index]
    axes[0].barh(agg.index, agg["stab"], color=colors)
    axes[0].set_xlabel("Mean bootstrap stability (ARI), K-Means k=3..8")
    axes[0].set_title("Stability by preprocessing variant (green = log1p)")
    axes[0].tick_params(axis="y", labelsize=8)
    for scaler, marker in (("standard", "o"), ("robust", "s")):
        for lg, col in ((True, PALETTE[2]), (False, PALETTE[3])):
            s = d[(d["scaler"] == scaler) & (d["log1p"] == lg)]
            axes[1].scatter(s["min_cluster_pct"], s["silhouette"], marker=marker, color=col, alpha=0.7, s=30, label=f"{'log1p' if lg else 'raw'} + {scaler}")
    axes[1].set_xscale("log")
    axes[1].set_xlabel("Smallest cluster (% of customers, log scale)")
    axes[1].set_ylabel("Silhouette")
    axes[1].set_title("High silhouette at tiny cluster sizes = outlier isolation, not structure")
    axes[1].legend(fontsize=8)
    return _save(fig, out)


def plot_elbow(grid: pd.DataFrame, out: Path) -> Path:
    km = grid[grid["algorithm"] == "kmeans"].sort_values("k_param")
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(km["k_param"], km["inertia"], "o-", color=PALETTE[0])
    drops = -km["inertia"].pct_change() * 100
    for k, y, dr in zip(km["k_param"], km["inertia"], drops):
        if not np.isnan(dr):
            ax.annotate(f"-{dr:.0f}%", (k, y), textcoords="offset points", xytext=(6, 6), fontsize=8)
    ax.set_xlabel("k")
    ax.set_ylabel("Inertia (within-cluster SSE)")
    ax.set_title("K-Means elbow (labels = % inertia drop vs. previous k)")
    return _save(fig, out)


def plot_metrics_by_k(grid: pd.DataFrame, out: Path) -> Path:
    """Silhouette / Davies-Bouldin / Calinski-Harabasz vs k for the k-parameterised algorithms."""
    d = grid[grid["k_param"].notna()].copy()
    d["series"] = d.apply(lambda r: r["algorithm"] + (f" ({r['params'].get('linkage')})" if r["algorithm"] == "agglomerative" else "") + (f" ({r['params'].get('covariance_type')})" if r["algorithm"] == "gmm" else ""), axis=1)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, (col, title) in zip(axes, [("silhouette", "Silhouette (higher better)"), ("davies_bouldin", "Davies-Bouldin (lower better)"), ("calinski_harabasz", "Calinski-Harabasz (higher better)")]):
        for i, (name, g) in enumerate(d.groupby("series")):
            g = g.sort_values("k_param")
            ax.plot(g["k_param"], g[col], marker="o", ms=3, label=name, color=sns.color_palette("tab20")[i % 20])
        ax.set_title(title)
        ax.set_xlabel("k")
    axes[2].legend(fontsize=7, bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.suptitle("Internal metrics vs. k (average-linkage rows are inflated by singleton clusters)", y=1.02)
    return _save(fig, out)


def plot_gmm_bic(grid: pd.DataFrame, out: Path) -> Path:
    g = grid[grid["algorithm"] == "gmm"].copy()
    g["cov"] = g["params"].apply(lambda p: p["covariance_type"])
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for cov, s in g.groupby("cov"):
        s = s.sort_values("k_param")
        axes[0].plot(s["k_param"], s["bic"], marker="o", label=cov)
        axes[1].plot(s["k_param"], s["aic"], marker="o", label=cov)
    axes[0].set_title("GMM BIC (lower better)")
    axes[1].set_title("GMM AIC (lower better)")
    for ax in axes:
        ax.set_xlabel("components")
    axes[0].legend()
    return _save(fig, out)


def plot_density_models(grid: pd.DataFrame, out: Path) -> Path:
    d = grid[grid["algorithm"].isin(["dbscan", "hdbscan"])]
    fig, ax = plt.subplots(figsize=(8, 5))
    for algo, marker, col in (("dbscan", "o", PALETTE[1]), ("hdbscan", "s", PALETTE[4])):
        s = d[d["algorithm"] == algo]
        ax.scatter(s["noise_pct"], s["max_cluster_pct"], marker=marker, color=col, s=60, alpha=0.8, label=algo)
    ax.axvline(20, color="k", ls="--", lw=1)
    ax.text(20.5, 5, "max noise 20%", fontsize=8)
    ax.set_xlabel("Noise (% of customers labelled -1)")
    ax.set_ylabel("Largest cluster (% of customers)")
    ax.set_title("Density-based models: one dominant cluster and/or heavy noise")
    ax.legend()
    return _save(fig, out)


def plot_model_comparison(table: pd.DataFrame, out: Path, selected_id: str) -> Path:
    t = table.copy()
    t["label"] = t["model_id"].str.replace("n_init=10", "").str.replace("batch_size=1024,", "").str.replace(",n_init=3", "").str.strip(",] ").str.replace(",]", "]")
    t = t.sort_values(["in_pool", "mean_rank"], ascending=[True, False])
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
    colors = [PALETTE[2] if m == selected_id else (PALETTE[0] if ip else "#bbbbbb") for m, ip in zip(t["model_id"], t["in_pool"])]
    for ax, (col, title) in zip(axes, [("silhouette", "Silhouette"), ("davies_bouldin", "Davies-Bouldin (lower better)"), ("stability_ari_mean", "Stability (ARI)")]):
        ax.barh(t["label"], t[col], color=colors, xerr=t["stability_ari_std"] if col == "stability_ari_mean" else None)
        ax.set_title(title)
    axes[0].tick_params(axis="y", labelsize=7)
    fig.suptitle("Finalists: green = selected, blue = in ranking pool, grey = rejected (see table)", y=1.0)
    return _save(fig, out)


# ------------------------------------------------------------ clusters / profiles
def _segment_colors(names: dict[int, str]) -> dict[int, tuple]:
    return {c: PALETTE[i % 10] for i, c in enumerate(sorted(names))}


def plot_cluster_sizes(summary: pd.DataFrame, out: Path) -> Path:
    s = summary.sort_values("cluster")
    cols = [PALETTE[i % 10] for i in range(len(s))]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars = ax.bar(s["segment_name"], s["n_customers"], color=cols)
    for b, p in zip(bars, s["pct_customers"]):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height(), f"{int(b.get_height()):,}\n({p:.1f}%)", ha="center", va="bottom", fontsize=9)
    ax.set_ylim(0, s["n_customers"].max() * 1.18)
    ax.set_ylabel("Customers")
    ax.set_title("Segment sizes")
    plt.setp(ax.get_xticklabels(), rotation=15, ha="right")
    return _save(fig, out)


def plot_pca(xy: np.ndarray, labels: np.ndarray, names: dict[int, str], explained: np.ndarray, out: Path) -> Path:
    fig, ax = plt.subplots(figsize=(8.5, 6.5))
    colors = _segment_colors(names)
    for c in sorted(names):
        m = labels == c
        ax.scatter(xy[m, 0], xy[m, 1], s=5, alpha=0.45, color=colors[c], label=f"{names[c]} (n={m.sum():,})")
    ax.set_xlabel(f"PC1 ({explained[0] * 100:.1f}% variance)")
    ax.set_ylabel(f"PC2 ({explained[1] * 100:.1f}% variance)")
    ax.set_title(f"Segments in PCA space (PC1+PC2 = {explained[:2].sum() * 100:.1f}% of variance; overlap in 2-D is expected)")
    ax.legend(markerscale=3, fontsize=8)
    return _save(fig, out)


def plot_profile_heatmap(index_df: pd.DataFrame, names: dict[int, str], out: Path, features: list[str]) -> Path:
    d = index_df[features].copy()
    d.index = [names[c] for c in d.index]
    fig, ax = plt.subplots(figsize=(14, 0.9 * len(d) + 2.5))
    sns.heatmap(d / 100, cmap="RdBu_r", center=1, vmin=0, vmax=3, annot=d.round(0).astype(int), fmt="d", annot_kws={"size": 8}, linewidths=0.4, ax=ax, cbar_kws={"label": "Index vs. population mean (1.0 = average)"})
    ax.set_title("Segment profile index (segment mean / population mean x 100)")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    return _save(fig, out)


def plot_segment_bars(summary: pd.DataFrame, out: Path) -> Path:
    s = summary.sort_values("cluster")
    metrics = [
        ("BALANCE_median", "Median balance"), ("PURCHASES_median", "Median purchases"), ("CASH_ADVANCE_median", "Median cash advance"),
        ("CREDIT_LIMIT_median", "Median credit limit"), ("PAYMENTS_median", "Median payments"), ("BALANCE_TO_LIMIT_median", "Median utilisation"),
        ("PURCHASES_FREQUENCY_median", "Median purchase frequency"), ("pct_full_payers_ge_50", "% paying in full >= 50% of the time"),
    ]
    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    cols = [PALETTE[i % 10] for i in range(len(s))]
    for ax, (col, title) in zip(axes.ravel(), metrics):
        ax.bar(range(len(s)), s[col], color=cols)
        ax.set_title(title, fontsize=10)
        ax.set_xticks(range(len(s)))
        ax.set_xticklabels([n.replace(" ", "\n", 1) for n in s["segment_name"]], fontsize=7)
    fig.suptitle("Segment comparison on observed behaviour", y=1.0, fontsize=13)
    return _save(fig, out)


# ------------------------------------------------------------- extra EDA / preprocessing / PCA
def plot_outlier_share(audit: pd.DataFrame, n_rows: int, out: Path) -> Path:
    """Share of customers beyond the 3xIQR upper fence, per column (no rows are removed)."""
    d = audit.assign(pct=audit["n_outliers_3iqr"] / n_rows * 100).sort_values("pct")
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(d["column"], d["pct"], color=PALETTE[1])
    for y, v in enumerate(d["pct"]):
        ax.text(v, y, f" {v:.1f}%", va="center", fontsize=8)
    ax.set_xlabel("% of customers above Q3 + 3 x IQR")
    ax.set_xlim(0, d["pct"].max() * 1.15)
    ax.set_title("How many customers are extreme on each feature (kept, not deleted)")
    return _save(fig, out)


def plot_transform_steps(df: pd.DataFrame, cols: list[str], log_cols: list[str], out: Path) -> Path:
    """Raw -> log1p -> robust-scaled histograms for selected features."""
    from sklearn.preprocessing import RobustScaler

    fig, axes = plt.subplots(3, len(cols), figsize=(3.2 * len(cols), 8))
    for j, c in enumerate(cols):
        raw = df[c].astype(float)
        step2 = np.log1p(raw) if c in log_cols else raw
        scaled = RobustScaler().fit_transform(step2.to_frame()).ravel()
        for i, (v, lab) in enumerate([(raw, "raw"), (step2, "log1p" if c in log_cols else "unchanged (proportion)"), (scaled, "robust-scaled")]):
            ax = axes[i, j]
            ax.hist(v, bins=50, color=PALETTE[i])
            ax.set_title(f"{c}\n{lab} | skew {pd.Series(v).skew():.2f}", fontsize=8)
            ax.tick_params(labelsize=7)
    fig.suptitle("Preprocessing steps: raw -> log1p -> robust scaling", y=1.0, fontsize=13)
    return _save(fig, out)


def plot_pca_scree(explained: np.ndarray, out: Path) -> Path:
    cum = np.cumsum(explained)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(np.arange(1, len(explained) + 1), explained * 100, color=PALETTE[0], label="per component")
    ax.plot(np.arange(1, len(explained) + 1), cum * 100, "o-", color=PALETTE[3], label="cumulative")
    for thr in (80, 90):
        ax.axhline(thr, color="grey", ls="--", lw=0.8)
        k = int(np.argmax(cum * 100 >= thr)) + 1
        ax.text(len(explained) * 0.55, thr + 1, f"{thr}% reached at {k} components", fontsize=8)
    ax.axvline(2.5, color="k", ls=":", lw=0.8)
    ax.text(2.6, 50, f"first 2 PCs = {cum[1] * 100:.1f}%", fontsize=8)
    ax.set_xlabel("Principal component")
    ax.set_ylabel("% of variance")
    ax.set_title("PCA variance explained (a 2-D plot cannot show everything)")
    ax.legend(loc="center right")
    return _save(fig, out)


def plot_pca_other_views(xyz: np.ndarray, labels: np.ndarray, names: dict[int, str], explained: np.ndarray, out: Path) -> Path:
    colors = _segment_colors(names)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    for ax, (a, b) in zip(axes, [(0, 2), (1, 2)]):
        for c in sorted(names):
            m = labels == c
            ax.scatter(xyz[m, a], xyz[m, b], s=4, alpha=0.4, color=colors[c], label=names[c])
        ax.set_xlabel(f"PC{a + 1} ({explained[a] * 100:.1f}%)")
        ax.set_ylabel(f"PC{b + 1} ({explained[b] * 100:.1f}%)")
    axes[0].legend(markerscale=3, fontsize=8)
    fig.suptitle("Other PCA views: PC3 separates Cash-Advance Borrowers from Active Balance-Carriers", y=1.0)
    return _save(fig, out)
