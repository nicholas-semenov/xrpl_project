# Project Progress — XRPL AML Scoring

## Current status
- Stage: Prediction. A validated two-stage AML model has produced a checked test CSV.
- Current objective: Improve AP on decoy look-alike scenarios and on hard relay positives; optionally package the submission ZIP.
- Last verified working outcome (2026-10-24): `scripts/aml_model.py` wrote `outputs/aml_two_stage_v1/predictions.csv` (40,101 rows, all checks pass, byte-identical on repeat); 34/34 tests pass.

## Completed and verified
- Earlier work (data audit, canonical payment preparation, wallet graphs, starter logistic baseline) was reproduced on this machine with identical metrics, hashes and reports.
- `feature_workspace/aml_features.py`: 198 label-free features per payment, about 15 s per split. Groups: transaction, wallet/pair, 10m–24h windows, pass-through, chains, cycles, scatter-gather, butterflies, burst clusters and tempo. Outputs go to `generated/aml_features_v1/{split}/`.
- `feature_workspace/score_context.py`: max/mean of neighbours' stage-1 scores (same-wallet events, chain neighbours, burst members), excluding the payment itself.
- `scripts/aml_model.py`: grouped CV, two wallet stress tests, final fit, test inference and a strict CSV write/re-read. Writes `reports/aml_model_metrics.json`.
- `tests/test_aml_features.py` (8 tests): causal cycles/chains, fans, burst scales, order/label-column invariance, issuer roles, context self-exclusion, CSV round trip.

## Decisions and assumptions
- Whole-week, label-free context per split (allowed by the dataset README). Splits are never mixed; issuer addresses differ per split, so issuers are coded by role (GBP issuer / EUR issuer).
- Primary validation: StratifiedGroupKFold(5, seed 0) grouped by `scenario_id` (scenarios are label-pure). Folds and diagnostics are the only use of annotations.
- Stage-2 context for labelled rows uses nested inner-OOF stage-1 scores; for test rows it uses a train-fitted stage-1 model.
- Removed after 3-seed ablation (no gain; transfer risk): fee drops, amount decimals, hour of day. Burst-cluster features are the strongest group: removing them costs about 0.035 AP.
- HistGradientBoosting kept: LightGBM was within ±0.01 AP, and keeping HistGradientBoosting adds no dependency (LightGBM was uninstalled again).
- Model selection used the same grouped CV, so that figure is mildly optimistic.

## Results and checks
- Grouped CV: stage 1 AP 0.5792; **stage 2 AP 0.5942**; ROC-AUC 0.8963; top-1% precision 0.871 (350/402); fold AP 0.573–0.661.
- Baseline wallet split, full-week features: AP 0.4600. The like-for-like isolated version (features recomputed per subset) gives **0.2592**, against the starter baseline's 0.0836 on the same 10,112 rows.
- Median positive rank percentile: easy 0.987, medium 0.922, hard 0.619. Relay hard positives are near chance.
- Test predictions: 40,101 distinct scores in [5.9e-5, 0.998]; 1.83% above 0.5 (train OOF 2.04%); SHA-256 a73fdac868e6f70702611c813972e29ad061d760c359a08de87b78c874d1a41f.
- Working commands (verified): `.venv/bin/python -m feature_workspace.prepare_transactions`; `.venv/bin/python -m feature_workspace.build_wallet_graph`; `.venv/bin/python -m feature_workspace.aml_features`; `.venv/bin/python scripts/aml_model.py` (~5.5 min; `--skip-validation` ~1 min); `.venv/bin/python scripts/baseline.py`; `.venv/bin/python -m unittest discover -s tests` (34 OK).
- Environment on this machine: conda env `xrpl-py310` (Python 3.10.15) → `.venv` with `requirements-baseline.txt`. The system python3 is 3.14 and must not be used.

## Git synchronization
- Branch/upstream: master / origin/master. The session-start `git pull --ff-only` fast-forwarded 62730a8 → 9faef27.
- The user had asked to skip routine git steps, then asked to publish the project: the AML-model work was committed on master and pushed, with full history (including data/), to the new remote `xrpl_project` (https://github.com/nicholas-semenov/xrpl_project.git) on 2026-10-24. `origin` (xrpl) was not updated.

## Blockers / risks
- Test "batch variations absent from training" (dataset README) may shift feature distributions. Not measurable without a leaderboard upload, and nothing has been uploaded.
- Remaining errors: decoy motif scenarios (bipartite/chain/cycle have the lowest positive-vs-decoy AUC, 0.75–0.86), and background payments that touch a motif wallet mid-burst.
- Relay hard positives (200 of 1,250) show no signal yet; the 25 ambiguous positives are unpredictable by design.

## Next actions (in priority order)
1. Scenario-level reconstruction features (motif tempo/value retention per burst) to separate decoys from planted motifs; check with 3-seed grouped CV.
2. Investigate relay hard cases using longer-history aggregates per hub (counterparty change across rounds).
3. Package the code ZIP (README, requirements, LICENSE, attribution list) if the user wants to submit.

## Important paths
- feature_workspace/aml_features.py, feature_workspace/score_context.py, scripts/aml_model.py, tests/test_aml_features.py
- reports/aml_model_metrics.json — all measured metrics, parameters, versions, hashes.
- outputs/aml_two_stage_v1/{predictions.csv, train_oof_predictions.csv, models.joblib, metrics.json} (git-ignored).
- feature_workspace/generated/{transaction_tables_v1, wallet_graphs_v1, aml_features_v1}/ (git-ignored).
- scripts/baseline.py, reports/baseline_metrics.json — preserved starter baseline.
