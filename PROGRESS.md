# Project Progress — XRPL AML Scoring

## Current status
- Stage: Prediction (iteration 2). v1 scored 64.11 on the leaderboard; v2 (local-structure features) is validated and ready to upload.
- Current objective: Upload v2 and compare; further gains need medium positives and decoy separation (hard relays look undetectable by design).
- Last verified working outcome (2026-10-24): `scripts/aml_model.py` wrote `outputs/aml_two_stage_v2/predictions.csv` (40,101 rows, all checks pass, byte-identical on repeat and when rebuilt from `outputs/submission/submission_code_v2.zip`); 36/36 tests pass.

## Completed and verified
- v1 (commit 68a3d25): 198 features plus a stage-2 score context. Leaderboard **64.11** (top-1% precision 95%, recall 30%; easy 50%, medium 10%, observable hard 0%, ambiguous 0% inside the top 1%).
- v2: +45 local-structure features in `feature_workspace/aml_features.py` (`_structure_features`): common neighbours/Jaccard, ego-subgraph edges/density/tempo/conversion (30m–6h), and flow clusters of amount-consistent forwards (30m/2h/6h). `score_context.py` adds ego-1h and flow-2h neighbour contexts. 243 features in `generated/aml_features_v2/`.
- `scripts/aml_model.py` runs outer folds and the two stress tests in parallel processes (joblib, 2 threads each). Fold metrics are identical to the sequential run; the full run takes about 3.3 min.
- Submission ZIPs: `outputs/submission/submission_code.zip` (v1), `outputs/submission/submission_code_v2.zip` (0.12 MiB, 34 files, with LICENSE (MIT) and ATTRIBUTION.md).
- Tests: `tests/test_aml_features.py` now has 10 tests (adds dense-motif-vs-star ego and flow-cluster tests).

## Decisions and assumptions
- Whole-week, label-free context per split; issuers are coded by role, since addresses differ per split.
- Primary validation: StratifiedGroupKFold(5) grouped by `scenario_id`; stage-2 context uses nested inner-OOF scores.
- Kept via 3-seed ablation: burst clusters (−0.035 if removed), ego/common-neighbour features (−0.033), flow clusters (−0.013). Removed: fee, decimals, hour of day, ego isolation/amount CV/sender and receiver counts.
- HistGradientBoosting defaults kept: 63 leaves, LightGBM, lower LR and feature subsampling were all within ±0.01.
- Stage 1 and stage 2 are about tied on v2 grouped CV (seed 0: 0.616/0.608; seed 1: 0.617/0.629); stage 2 is kept because it is better on the stress tests and was used for 64.11.
- Hard relay: 12 cohorts × 100 template-matched scenarios, exactly 1 planted per cohort; no scenario aggregate (growth, counterparty Jaccard, degrees, timing, values) consistently separates it.

## Results and checks
- v2 grouped CV: stage 1 0.6160, **stage 2 0.6076** (v1 0.5942); ROC-AUC 0.9002; top-1% precision 0.853. Fold APs 0.564–0.725.
- v2 wallet stress, full-week features: **0.5053** (v1 0.4600). Isolated features: **0.2979** (v1 0.2592; starter 0.0836).
- v2 test predictions: SHA-256 3ac342c6885401170c4c8fc7a62fca1a336f2b89793c2cc326d23ceeb6f5cbe2; 1.76% of scores above 0.5. Spearman vs v1 0.82; top-402 overlap 338.
- Working commands: `.venv/bin/python -m feature_workspace.prepare_transactions`; `.venv/bin/python -m feature_workspace.aml_features --run-name aml_features_v2`; `.venv/bin/python scripts/aml_model.py` (`--skip-validation` ~1 min); `.venv/bin/python -m unittest discover -s tests` (36 OK).
- Environment: conda env `xrpl-py310` (Python 3.10.15) → `.venv` with `requirements-baseline.txt`. The system python3 is 3.14 and must not be used.

## Git synchronization
- The user handles git: they pushed 68a3d25 plus their own "test" commits (through ceb51e9) to `xrpl_project`, and trimmed `.gitignore` to `.venv/`. The v2 changes are uncommitted; per user instruction the agent does not commit or push.

## Blockers / risks
- Test "batch variations absent from training" (dataset README) may shift feature distributions. Not measurable without a leaderboard upload, and nothing has been uploaded.
- Remaining errors: decoy motif scenarios (bipartite/chain/cycle have the lowest positive-vs-decoy AUC, 0.75–0.86), and background payments that touch a motif wallet mid-burst.
- Relay hard positives (200 of 1,250) show no signal yet; the 25 ambiguous positives are unpredictable by design.

## Next actions (in priority order)
1. Upload `outputs/aml_two_stage_v2/predictions.csv` and `outputs/submission/submission_code_v2.zip`; record the score here.
2. Medium positives (median rank 0.93): slow motifs (15–28 min gaps); try scenario-like grouping at larger scales and stage-1/stage-2 blending, checked with 2–3 seeds in parallel.
3. Decoy vs planted motifs: compare per-family tempo/value statistics of the reconstructed structures.

## Important paths
- feature_workspace/aml_features.py, feature_workspace/score_context.py, scripts/aml_model.py, tests/test_aml_features.py
- reports/aml_model_metrics.json (v1), reports/aml_model_metrics_v2.json (v2) — measured metrics, parameters, versions, hashes.
- outputs/aml_two_stage_v1/ (64.11 submission) and outputs/aml_two_stage_v2/ — predictions, OOF, models, metrics; outputs/submission/ — code ZIPs.
- feature_workspace/generated/{transaction_tables_v1, wallet_graphs_v1, aml_features_v1}/ (git-ignored).
- scripts/baseline.py, reports/baseline_metrics.json — preserved starter baseline.
