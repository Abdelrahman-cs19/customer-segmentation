# Credit-Card Customer Segmentation

End-to-end unsupervised segmentation of 8,950 credit-card customers: audit, cleaning, feature engineering, a reproducible scikit-learn preprocessing pipeline, six clustering algorithms compared on multiple metrics and stability, segment profiling, a Streamlit dashboard and a full report.

**Result:** K-Means with **k = 4** on 19 log-transformed, robust-scaled features (silhouette 0.277, Davies-Bouldin 1.18, Calinski-Harabasz 3,695, bootstrap stability ARI 0.993, smallest segment 13.3%). The equal-weight composite rank actually preferred k = 3; k = 4 was chosen by a **documented manual override** made after reviewing the profiles (see [Model selection](#model-selection)).

## Dataset

`data/raw/Customer-Data - 2.csv`: 8,950 customers x 18 columns (`CUST_ID` plus 17 numeric fields: balance, purchases and their one-off/installment split, cash advance, frequencies, transaction counts, credit limit, payments, minimum payments, full-payment share, tenure). The raw file is never modified (read-only, SHA-256 in `data/raw/SHA256SUM.txt`, re-verified every run). Check the dataset's licence before redistributing it.

## Key data findings

- No duplicates or negatives. Missing: `MINIMUM_PAYMENTS` 313 (3.5%), `CREDIT_LIMIT` 1.
- `PURCHASES` equals `ONEOFF + INSTALLMENTS` in 99.78% of rows, so it is excluded from the model to avoid double counting.
- `MINIMUM_PAYMENTS` is **not missing at random**: all 240 zero-payment customers have it missing (plus 73 customers who did pay).
- 8 rows have `CASH_ADVANCE_FREQUENCY` > 1 (invalid for a frequency); clipped to 1.
- Heavy right skew and long tails of legitimate high-value customers: **no rows were removed**.

## Preprocessing decisions

| Decision | Choice | Evidence |
|---|---|---|
| Missing `MINIMUM_PAYMENTS` | flag + median | Re-clustering with zero-fill or balance-scaled fill gives ARI 0.99 vs median (`outputs/imputation_sensitivity.csv`) |
| Skew | `log1p` on non-negative, unbounded, skew > 1 columns | Non-log variants isolated clusters of ~2 customers and had lower stability |
| Scaling | `RobustScaler` | Better silhouette and Davies-Bouldin than `StandardScaler` at every k; highest stability |
| Redundancy | exclude `PURCHASES`; keep `*_TRX` | Dropping the two `*_TRX` counts did not improve stability |
| Features | add utilisation, payment-vs-minimum, one-off share, avg ticket | Tie on metrics with the base set; chosen for repayment/utilisation signal |
| `TENURE` | profiled, not clustered | 85% of customers are at 12; reflects window, not behaviour |

The full 16-variant comparison is in `outputs/preprocessing_comparison.csv`. Note the feature-set choice was a statistical tie on metrics, yet K-Means on the base set agrees with the final labels only at ARI 0.64, so individual assignments are less certain than the segment archetypes.

## Algorithms

93 candidates: K-Means, MiniBatch K-Means, Agglomerative (ward, average), Gaussian Mixture (4 covariance types), DBSCAN and HDBSCAN, all with fixed seeds. Hard constraints for eligibility: 3-8 clusters, smallest cluster >= 2%, noise <= 20% (42 of 93 pass). Finalists get a subsample-stability test (10 rounds x 80% subsamples, ARI).

Why several were rejected (actual outcomes): average linkage scores well but isolates single customers; DBSCAN collapses or leaves 89-99% of customers in one cluster; HDBSCAN labels 10-52% as noise; full-covariance GMM has silhouettes of 0.03-0.14.

### Model selection

Top finalists (full table: `outputs/model_comparison.csv`):

| Model | Silhouette | DB | CH | Stability ARI | Mean rank |
|---|---|---|---|---|---|
| K-Means k=3 | 0.396 | 1.035 | 3,833 | 0.995 | 1.25 (composite winner) |
| **K-Means k=4 (selected)** | 0.277 | 1.184 | 3,695 | 0.993 | 4.50 |
| GMM spherical k=4 | 0.263 | 1.157 | 3,468 | 0.995 | 4.75 |

k = 3 has the best composite rank, but the composite is biased toward small k. Its largest segment (71%) mixes customers who never purchase but take cash advances with active purchasers; k = 4 splits exactly that group, leaves the other two segments nearly unchanged, sits at the inertia elbow and is equally stable. This override was decided after viewing profiles, not pre-registered. To use k = 3, edit `selection.final_model_id` in `configs/config.yaml`.

## Segments (observed behaviour; medians)

| Segment | Customers | Balance | Purchases | Purchase freq. | Cash advance | Credit limit | Payments | Pay in full >= 50% |
|---|---|---|---|---|---|---|---|---|
| Cash-Advance Borrowers | 1,958 (21.9%) | 1,754 | 0 | 0.00 | 1,366 | 3,000 | 820 | 0% |
| Low-Activity Light Users | 1,193 (13.3%) | 23 | 143 | 0.17 | 0 | 2,700 | 326 | 8% |
| Frequent Full-Payers | 1,289 (14.4%) | 90 | 916 | 0.92 | 0 | 4,000 | 1,100 | 91% |
| Active Balance-Carriers | 4,510 (50.4%) | 1,180 | 677 | 0.75 | 0 | 3,500 | 969 | 0% |

Profitability and credit-risk statements in the report are **assumptions**: the dataset has no revenue, interest, fee or delinquency fields. Marketing suggestions per segment are in `reports/final_report.md` (Section 8).

## Limitations

Single snapshot (no time dimension); internal metrics favour compact clusters and small k; behaviour is a continuum so segments are useful partitions rather than natural groups; silhouette is not strictly comparable across feature spaces; segment-naming thresholds are tuned to this data; a score is the closest of four centroids, not a probability. See report Section 10.

## Project layout

```
configs/config.yaml        all parameters (seed, features, grids, constraints, override)
src/segmentation/          config, logging, data, cleaning, features, preprocessing, clustering,
                           evaluation, selection, experiments, profiling, visuals, pipeline, reporting
app/streamlit_app.py       dashboard
tests/                     pytest suite
run_pipeline.py            entry point
data/raw/                  original data (read-only)
outputs/                   model_comparison.csv, customer_assignments.csv, segment_summary.csv, ...
figures/                   EDA, evaluation and segment figures
reports/final_report.md    detailed report
```

## Installation and execution

Tested on Python 3.12.

```bash
pip install -r requirements-dev.txt         # runtime + pytest (requirements.txt = runtime only)
python run_pipeline.py                      # ~8-12 min on one CPU core; rebuilds outputs/, figures/, models/
PYTHONPATH=src python -m segmentation.reporting   # regenerates reports/final_report.md
streamlit run app/streamlit_app.py          # dashboard at http://localhost:8501
pytest
```

`python run_pipeline.py --reuse-prep-comparison` skips the ~3.5 min preprocessing study if `outputs/preprocessing_comparison.csv` exists. The dashboard only needs the committed `outputs/`, `figures/` and `models/final_model.joblib`; it does not re-run the pipeline.

## Dashboard

Nine tabs: **Overview** (sizes, PCA, scree), **Segment profiles**, **EDA** (9 figures + audit tables), **Preprocessing** (pipeline, transform steps, 16-variant comparison, imputation sensitivity), **Test the model**, Feature explorer, Customer lookup, Model comparison, Downloads.

**Test the model** has two modes:
- *Single customer*: enter the 15 raw features (start from any segment's median customer or a blank one). You get the assigned segment, the distance to each centroid, an assignment margin and a comparison with the segment median.
- *Batch*: upload a CSV (template provided; up to 20,000 rows), download the scored file.

Scoring applies exactly the training-time cleaning (stored fill values, clipping), feature engineering, preprocessing and K-Means. The margin (1 - nearest/second-nearest centroid distance) is a closeness heuristic, **not a probability**, and a segment is a behavioural group, not a credit decision. Inputs far outside the training range are flagged as extrapolations.

## Deploy to the web (Streamlit Community Cloud)

I could not deploy this for you (it needs your GitHub and Streamlit accounts). The project is deployment-ready:

1. Create a GitHub repository and push this folder:
   ```bash
   git init && git add . && git commit -m "Customer segmentation" 
   git branch -M main && git remote add origin https://github.com/<you>/<repo>.git && git push -u origin main
   ```
2. Go to <https://share.streamlit.io>, click **Create app**, choose the repo and branch `main`, and set **Main file path** to `app/streamlit_app.py`.
3. Under **Advanced settings** choose **Python 3.12** (the pinned library versions were tested on it; the saved model is a pickle that must load with the same scikit-learn version).
4. Deploy. The first build takes a few minutes.

Before you do, consider:
- **Privacy:** the batch upload sends data to the host. Do not upload real customer data to a public app; use a private app (Streamlit's sharing settings) or add authentication.
- **Dataset licence:** `data/raw/` is tracked so the pipeline is reproducible. The dashboard does not read it; remove it from the repo if you may not redistribute it (the artifact tests that need it will then skip or fail).
- Other hosts (Hugging Face Spaces, Render, a Docker container) should work with the same requirements but were **not tested**.

## Verification status (what was actually executed)

- Full pipeline run completed (452 s with the cached preprocessing study; that study was produced by the same code in an earlier run of this pipeline). Results were identical across the repeated runs (same model, same segment sizes).
- `pytest`: **78 passed**, including: scoring all 8,950 raw customers with the saved model reproduces the pipeline's assignments exactly; ID/label alignment; raw-data checksum; a headless run of the dashboard including the "Test the model" form.
- Deployment simulation: a `git` checkout of only the tracked files, a fresh virtual environment with `requirements.txt` only, the app run headlessly (no exceptions, scoring works), and the real Streamlit server started (`/_stcore/health` = ok, page HTTP 200).
- **Not done:** deployment to an actual hosting service, and viewing the dashboard in a real browser.
#   c u s t o m e r - s e g m e n t a t i o n  
 