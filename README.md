# XRPL AML scoring

## Current model: two-stage AML scorer (`aml_two_stage_hgb_v1`)

Produce the test predictions CSV from the supplied data (Python 3.10.15, `requirements-baseline.txt`; about 6 minutes and under 2 GB RAM on an Apple M-series arm64 laptop, of which validation is about 5). First unpack the organiser packages so that `data/traning_data/` holds the training package (`train/`, `validation/`, `payments.py`) and `data/test_data/` holds the four test files. Licence: [MIT](LICENSE); third-party code, libraries and AI assistance are listed in [ATTRIBUTION.md](ATTRIBUTION.md).

```sh
python3.10 -m venv .venv
.venv/bin/python -m pip install --no-cache-dir -r requirements-baseline.txt
.venv/bin/python -m feature_workspace.prepare_transactions   # canonical payments, all splits
.venv/bin/python -m feature_workspace.aml_features           # 198 label-free features per payment
.venv/bin/python scripts/aml_model.py                        # validate, fit, write predictions
# writes outputs/aml_two_stage_v1/predictions.csv (TRANSACTION_HASH,score; 40,101 rows)
# add --skip-validation to only fit and predict (~1 minute)
```

Method: stage 1 is a `HistGradientBoostingClassifier` (learning rate 0.05, 400 iterations, 31 leaves, min leaf 20, L2 1.0, seed 0) on the [AML features](feature_workspace/README.md#aml-behaviour-flow-and-motif-features). These cover wallet and pair behaviour, windowed pass-through flows, amount-consistent temporal chains, cycles, scatter-gather, bipartite butterflies and multi-scale burst clusters. Stage 2 is the same estimator with extra features: the max and mean stage-1 scores of each payment's graph neighbours (same-wallet events, chain neighbours, burst-cluster members). For labelled rows those stage-1 scores are nested out-of-fold, so no context feature contains a score fitted on that row's own label. Test features and contexts are computed only from the test week. The whole supplied window is used, as the dataset README allows (offline investigation, not causal live detection). Labels and scenario/cohort IDs never enter features. Scenario IDs only define CV folds.

Measured on labelled train only (`reports/aml_model_metrics.json`):

| Protocol | Stage-1 AP | Stage-2 AP | ROC-AUC | Top-1% precision |
|---|---:|---:|---:|---:|
| Scenario-grouped 5-fold CV, full train week (primary; closest to test conditions) | 0.5792 | **0.5942** | 0.8963 | 0.871 (350/402) |
| Baseline wallet split, full-week features | 0.4436 | 0.4600 | 0.8356 | 0.735 (75/102) |
| Baseline wallet split, features recomputed in isolated fit/holdout subsets (like-for-like with the baseline's 0.0836) | 0.2526 | **0.2592** | 0.7789 | 0.549 (56/102) |

Stage-2 fold APs range from 0.573 to 0.661 (mean 0.602, std 0.031). Median positive rank percentile: easy 0.987, medium 0.922, hard 0.619. The hard relay families score near chance, and the 25 "ambiguous" positives are label-independent by design. Main remaining errors are planted look-alike (decoy) scenarios and background payments that touch a motif wallet during a burst. A repeat `--skip-validation` run reproduced a byte-identical predictions CSV (SHA-256 `a73fdac8…`). Multi-threaded and single-threaded runs gave identical fold metrics. Model selection (feature-group ablation and a HistGradientBoosting vs LightGBM comparison, all within ±0.01 AP) used the same grouped CV. That makes the grouped-CV figure somewhat optimistic; the wallet-isolated figure is a pessimistic stress test because 62% of rows are purged. No test leaderboard score has been measured.

The earlier starter baseline below is preserved unchanged for comparison.

# First baseline (starter logistic regression)

Feature preparation has a separate [workspace](feature_workspace/README.md). Keep source datasets in `data/` as inputs and write derived tables only under `feature_workspace/generated/`, using the checked paths in `feature_workspace/paths.py`.

Reconstruct transaction-level amounts/assets and conversion counts for all three splits with `.venv/bin/python -m feature_workspace.prepare_transactions`. See the [conversion summary](reports/transaction_preparation.md) for measured counts, asset pairs and sample calculations. This preparation is label-free and does not retrain the baseline.

Build separate directed wallet multigraphs with `.venv/bin/python -m feature_workspace.build_wallet_graph`. Lossless chronological edge lists, wallet statistics and manifests are saved under `feature_workspace/generated/wallet_graphs_v1/`. See the [graph report](reports/wallet_graphs.md) and [workspace documentation](feature_workspace/README.md) for statistics, schemas and reusable adjacency access. No AML detectors or ML models are added by this step.

The current implementation evaluates the supplied starter logistic regression using only `data/traning_data/train/`. It does not score the separate practice or test datasets. No standalone organiser `baseline.py` exists in the supplied repository: the reference implementation is `data/traning_data/starter_notebook.ipynb`, with `payments.py` for USD payment reconstruction. Our executable adaptation is `scripts/baseline.py`.

## Reproduce

Run from the repository root using Python 3.10.15:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install --no-cache-dir -r requirements-baseline.txt
.venv/bin/python -m pip check
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python scripts/baseline.py
```

The run takes approximately 2.5 seconds in the verified environment (Apple arm64, macOS 27.0.1; package installation excluded). Direct and transitive dependencies are pinned. The system SciPy 1.15.3 and a fresh 1.15.3 wheel both failed to load a PROPACK binary on this macOS; the isolated environment with SciPy 1.13.1 ran successfully. Use `.venv/bin/python`, rather than the system installation.

Outputs:

- `reports/baseline_metrics.json`: measured metrics, split counts, all 12 coefficients, hyperparameters, versions, source hashes, timings and artifact fingerprints; committed.
- `outputs/baseline_wallet_seed0/split.csv`: assignment of every labelled transaction to fit, holdout or purged, with purge reason.
- `outputs/baseline_wallet_seed0/holdout_predictions.csv`: exactly one finite score in `[0, 1]` per retained holdout transaction.
- `outputs/baseline_wallet_seed0/model.joblib`: fitted scaler and logistic regression, trained only on fit rows.
- `outputs/baseline_wallet_seed0/metrics.json`: local copy of the metrics report.

Generated row-level artifacts, the fitted model and the environment are ignored by Git; the script regenerates them. The model is an evaluation artifact, not a model fitted on all available training data. Saved scores are read back and checked for exact coverage, order and numeric round-trip. Reloaded model predictions are also checked.

## Fixed validation protocol

An ordinary transaction-level split would share wallet activity and scenarios. Inspection found one connected component containing all 700 customer wallets, so preserving the full graph while holding out entire components is impossible here.

Before fitting, we fixed this stricter protocol:

1. Sort the 700 customer wallets, permute them with `numpy.random.default_rng(0)`, and assign 350 to holdout and 350 to fit. Targets, family and difficulty are not used in this assignment.
2. Retain a payment only when both customer endpoints belong to the same side. Purge the 19,979 crossing payments.
3. Remove fit payments whose non-null `scenario_id` occurs in holdout (1,126 rows), then those whose non-null `cohort_id` occurs in holdout (3,965 more rows). Metadata is used only for exclusion and diagnostics, never for features, target encoding or model fitting. Null metadata values are not treated as one shared group.
4. Compute the starter features separately from retained fit payments and retained holdout payments. Excluded payments are unavailable to both contexts. No transaction, customer wallet, non-null scenario or cohort is shared between the two contexts; runtime checks enforce this.
5. Fit the scaler, balanced class weights and regression coefficients only on fit rows. Evaluate once on holdout with no hyperparameter or seed search. The repeat run checks reproducibility of this same model and split.

| Partition | Payments | Positives | Positive share |
|---|---:|---:|---:|
| Fit | 4,949 | 169 | 3.415% |
| Holdout | 10,112 | 262 | 2.591% |
| Purged | 25,070 | 819 | 3.267% |

Whole-window, label-free context within each side is permitted by the supplied README. This evaluates offline investigation; future-within-week forwarding features are not causal live-detection features. Raw liquidity-provider legs are not model inputs: the bundled helper supplies one effective customer-to-recipient edge, valued from delivered assets and hourly train reference prices.

## Algorithm and features

The algorithm follows the supplied notebook: `StandardScaler` followed by L2 `LogisticRegression(class_weight="balanced", C=1.0, solver="lbfgs", max_iter=1000, tol=1e-4, random_state=0)`. BLAS uses one thread for fitting and prediction; the verified fit converged in 17 iterations. Balanced probabilities are ranking scores and are not claimed to be calibrated probabilities.

| Features | Definition |
|---|---|
| `log_value` | `log1p` of delivered reference USD value |
| `sender_payments`, `receiver_payments` | Whole-context outgoing sender count and incoming receiver count |
| `receiver_forwards_1h`, `sender_received_1h_before` | Any outgoing payment by receiver in the following hour; any receipt by sender in the preceding hour |
| `sender_burst_10m`, `receiver_burst_10m` | Outgoing/incoming counts within ±10 minutes, including the current event |
| `receiver_forwards_similar_1h`, `sender_received_similar_1h` | Later outgoing / earlier incoming payment within one hour and ±10% of the current USD amount |
| `receiver_minutes_to_forward` | Minutes until the receiver's strictly later outgoing payment, capped/defaulted at 60 |
| `sender_pass_through`, `receiver_pass_through` | Whole-context outgoing/incoming USD ratio, clipped at 3; undefined ratios filled with zero |

The simple presence windows include equal-time events, matching the notebook; similar-amount and next-send features require strict time direction. Stable `(TIMESTAMP, TRANSACTION_HASH)` ordering preserves deterministic alignment. Hashes are not model features. Unit tests compare all feature values against the supplied notebook functions. No extra AML motif detector or additional estimator was implemented.

## Measured results

Run ID: `starter_logistic_wallet_seed0`. Results are local holdout measurements, not a test leaderboard score.

| Metric | Result |
|---|---:|
| Average Precision | **0.0835839760** |
| `100 × AP` | **8.3583976** |
| ROC-AUC | **0.7348955710** |
| Precision@1% | **15.6863%** |
| Recall@1% | **6.1069%** |
| Suspicious in top 1% | **16 / 102** |

The review budget is `ceil(0.01 × 10,112) = 102`, not the test set's 402. Ties are broken by transaction hash ascending. Recall uses all 262 holdout positives. AP and ROC-AUC use all holdout scores with standard scikit-learn tie handling. All 10,112 scores are distinct. Holdout prevalence is 2.591%, a useful reference level for AP.

## Reliability and limitations

The isolation checks provide strong protection against shared labels, wallets, scenarios and cohorts inflating the estimate. The six tests cover notebook feature parity, label/annotation exclusion, temporal boundaries, split isolation and invariance, scaler fit isolation, and metric/hash alignment including ties and invalid scores. Package consistency checks pass.

However, this is one fixed stress test, not an unbiased estimate of performance on a complete new seven-day simulation. Purging removes 62.47% of transactions and breaks paths and histories. Training has only 169 positives, no retained cohort-annotated payments, and just 12 hard positives. Holdout contains 159 easy, 86 medium and 17 hard positives, rather than the original difficulty proportions. Both sides span the same week; this does not test future calendar drift. No confidence interval or fold variability has been estimated. These limitations can depress or otherwise change AP, so do not claim that the strict result is a guaranteed lower bound on the competition score.

The model ranks better than a prevalence-level reference but misses most positives at the 1% review budget. The next priority is validating stability and representativeness while preserving scenario isolation, then diagnosing the existing model's errors. Any future feature work should use fixed comparisons; no advanced patterns or alternative models are part of this milestone.

For a repeat check without replacing the canonical artifacts:

```sh
.venv/bin/python scripts/baseline.py --output-dir outputs/baseline_repeat --report outputs/baseline_repeat/metrics.json
.venv/bin/python -c 'import json; a=json.load(open("reports/baseline_metrics.json")); b=json.load(open("outputs/baseline_repeat/metrics.json")); assert all(a[k] == b[k] for k in ["metrics", "split_protocol", "standardized_coefficients", "artifact_sha256"]); print("Identical split, scores, coefficients and metrics")'
```

The verification run reproduced identical split and prediction CSV hashes, coefficients and metrics. Wall-clock timings can vary. Reproducibility across different BLAS/platform versions may be numeric rather than bit-for-bit; source/data hashes and installed versions are recorded in the report.

Sources and assistance: organiser notebook and payment helper, NumPy, pandas, PyArrow, SciPy, scikit-learn, joblib and threadpoolctl. Implementation and analysis were assisted by OpenAI Codex (GPT-6), prompted to reproduce the first baseline, isolate related training examples and report real held-out metrics. No external transaction data, services, pretrained model or hidden labels were used; executable feature and model code fully defines the analytical method.
