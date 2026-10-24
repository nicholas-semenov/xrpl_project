"""Two-stage gradient-boosted AML scorer: validation, final fit and test predictions.

Run: .venv/bin/python scripts/aml_model.py   (after prepare_transactions and aml_features)

Stage 1 scores each payment from label-free AML features (feature_workspace/aml_features.py).
Stage 2 adds neighbourhood aggregates of stage-1 scores (feature_workspace/score_context.py).
Stage-1 scores fed to stage 2 for labelled rows are always nested out-of-fold, so no row's
context contains a score fitted on its own label or on any held-out label.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import platform
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedGroupKFold
from joblib import Parallel, delayed
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from feature_workspace.aml_features import compute_features  # noqa: E402
from feature_workspace.paths import output_path  # noqa: E402
from feature_workspace.score_context import ScoreContext  # noqa: E402
from scripts.baseline import evaluate, isolated_split, require, validate_predictions  # noqa: E402

TRAIN = ROOT / "data/traning_data/train"
FEATURE_RUN = "aml_features_v2"
TABLE_RUN = "transaction_tables_v1"
SEED = 0
N_FOLDS = 5
THREADS = None  # all cores; a repeat run checks scores do not depend on thread scheduling
N_JOBS = 5           # parallel processes for independent folds / stress tests
THREADS_PER_JOB = 2
MODEL_PARAMS = dict(learning_rate=0.05, max_iter=400, max_leaf_nodes=31, min_samples_leaf=20,
                    l2_regularization=1.0)
ANNOTATIONS = ("is_suspicious", "family", "difficulty", "scenario_id", "cohort_id")


def build_model(seed: int = SEED) -> HistGradientBoostingClassifier:
    """No subsampling or early stopping, so a fit is deterministic for fixed inputs."""
    return HistGradientBoostingClassifier(**MODEL_PARAMS, random_state=seed)


def scenario_groups(annotations: pd.DataFrame) -> np.ndarray:
    """Group by scenario_id; payments without one form singleton groups.

    Annotation metadata only defines folds. It never enters features or fitting.
    """
    return annotations.scenario_id.fillna("tx:" + annotations.index.to_series()).to_numpy()


def fit_predict(x_fit: pd.DataFrame, y_fit: np.ndarray, x_eval: pd.DataFrame) -> np.ndarray:
    require(set(np.unique(y_fit)) == {0, 1}, "Fit rows need both classes")
    return build_model().fit(x_fit, y_fit).predict_proba(x_eval)[:, 1]


def nested_stage1(x: pd.DataFrame, y: np.ndarray, groups: np.ndarray, fit_rows: np.ndarray,
                  inner_seed: int) -> np.ndarray:
    """Stage-1 score for every row of one world.

    Rows in fit_rows get inner out-of-fold scores; all other rows are scored by a model
    fitted on fit_rows only, matching how unlabeled test rows are scored.
    """
    scores = fit_predict(x.iloc[fit_rows], y[fit_rows], x)
    inner = StratifiedGroupKFold(N_FOLDS, shuffle=True, random_state=inner_seed)
    for a, b in inner.split(fit_rows, y[fit_rows], groups[fit_rows]):
        scores[fit_rows[b]] = fit_predict(x.iloc[fit_rows[a]], y[fit_rows[a]], x.iloc[fit_rows[b]])
    return scores


def two_stage_eval(x: pd.DataFrame, y: np.ndarray, groups: np.ndarray, context: ScoreContext,
                   fit_rows: np.ndarray, eval_rows: np.ndarray, inner_seed: int) -> tuple:
    stage1 = nested_stage1(x, y, groups, fit_rows, inner_seed)
    full = pd.concat([x, context.aggregate(pd.Series(stage1, index=x.index)).loc[x.index]], axis=1)
    stage2 = fit_predict(full.iloc[fit_rows], y[fit_rows], full.iloc[eval_rows])
    return stage1[eval_rows], stage2


def _limited(function, *args):
    """Run one parallel job with a bounded OpenMP/BLAS thread pool."""
    with threadpool_limits(limits=THREADS_PER_JOB):
        return function(*args)


def holdout_metrics(index: pd.Index, scores: np.ndarray, targets: pd.Series) -> dict:
    predictions = pd.DataFrame({"TRANSACTION_HASH": index, "score": scores})
    return evaluate(predictions, targets.loc[index])


def grouped_cv(x, y, groups, context, annotations) -> tuple[dict, np.ndarray]:
    """Primary protocol: scenario-grouped stratified 5-fold CV on the full train week."""
    oof1, oof2, folds = np.zeros(len(y)), np.zeros(len(y)), []
    outer = list(StratifiedGroupKFold(N_FOLDS, shuffle=True, random_state=SEED).split(x, y, groups))
    for fit_rows, eval_rows in outer:
        require(not set(groups[fit_rows]) & set(groups[eval_rows]), "Scenario shared across folds")
    # Outer folds are independent: run them in separate processes (each fit is deterministic).
    results = Parallel(n_jobs=N_JOBS)(
        delayed(_limited)(two_stage_eval, x, y, groups, context, fit_rows, eval_rows, SEED + 1 + k)
        for k, (fit_rows, eval_rows) in enumerate(outer))
    for k, ((fit_rows, eval_rows), (s1, s2)) in enumerate(zip(outer, results)):
        oof1[eval_rows], oof2[eval_rows] = s1, s2
        folds.append({"rows": int(len(eval_rows)), "positives": int(y[eval_rows].sum()),
                      "stage1_ap": holdout_metrics(x.index[eval_rows], oof1[eval_rows], annotations.is_suspicious)["average_precision"],
                      "stage2_ap": holdout_metrics(x.index[eval_rows], oof2[eval_rows], annotations.is_suspicious)["average_precision"]})
        print(f"fold {k}: {json.dumps(folds[-1])}", flush=True)
    rank = pd.Series(oof2, index=x.index).rank(pct=True)
    positives = annotations.is_suspicious.eq(1)
    report = {
        "protocol": "StratifiedGroupKFold(5, shuffle, seed 0) grouped by scenario_id; label-free "
                    "features from the full train week; nested inner 5-fold stage-1 scores",
        "stage1": holdout_metrics(x.index, oof1, annotations.is_suspicious),
        "stage2": holdout_metrics(x.index, oof2, annotations.is_suspicious),
        "folds": folds,
        "fold_stage2_ap_mean": float(np.mean([f["stage2_ap"] for f in folds])),
        "fold_stage2_ap_std": float(np.std([f["stage2_ap"] for f in folds])),
        "diagnostics_only_positive_median_rank_pct": {
            d: float(rank[positives & annotations.difficulty.eq(d)].median())
            for d in ("easy", "medium", "hard")},
    }
    return report, oof2


def wallet_stress(transactions, x, y, groups, context, annotations, recompute: bool,
                  modules: list[Path] = ()) -> dict:
    """Secondary protocol: the baseline's seed-0 wallet split with scenario/cohort purging.

    recompute=False keeps full-week features (as used for test); recompute=True rebuilds
    features and score contexts separately from the retained fit and holdout payments,
    matching the baseline's feature-isolation protocol exactly.
    """
    payments = transactions.rename(columns={"SENDER": "FROM_ADDRESS", "RECEIVER": "TO_ADDRESS"})
    split = isolated_split(payments, annotations.reset_index()).set_index("TRANSACTION_HASH")
    partition = split.partition.reindex(x.index).to_numpy()
    fit_rows, eval_rows = np.where(partition == "fit")[0], np.where(partition == "holdout")[0]
    if not recompute:
        stage1, stage2 = two_stage_eval(x, y, groups, context, fit_rows, eval_rows, SEED + 11)
        index = x.index[eval_rows]
    else:
        parts = {}
        for name, rows in (("fit", fit_rows), ("holdout", eval_rows)):
            subset = transactions.set_index("TRANSACTION_HASH").loc[x.index[rows]].reset_index()
            base = compute_features(subset)
            extras = extra_features(None, subset, list(modules))
            features = pd.concat([base, *extras], axis=1) if extras else base
            require(list(features.columns) == list(x.columns), "Stress features differ from training")
            parts[name] = (features, ScoreContext(subset))
        xf, cf = parts["fit"]
        xh, ch = parts["holdout"]
        yf = annotations.is_suspicious.loc[xf.index].to_numpy()
        gf = groups[x.index.get_indexer(xf.index)]
        fit_all = np.arange(len(xf))
        s1_fit = nested_stage1(xf, yf, gf, fit_all, SEED + 12)
        stage1 = fit_predict(xf, yf, xh)
        full_fit = pd.concat([xf, cf.aggregate(pd.Series(s1_fit, index=xf.index)).loc[xf.index]], axis=1)
        full_hold = pd.concat([xh, ch.aggregate(pd.Series(stage1, index=xh.index)).loc[xh.index]], axis=1)
        stage2 = fit_predict(full_fit, yf, full_hold)
        index = xh.index
    return {"feature_context": "fit/holdout subsets recomputed separately" if recompute
            else "full train week (label-free)",
            "fit_rows": int(len(fit_rows)), "holdout_rows": int(len(eval_rows)),
            "stage1": holdout_metrics(index, stage1, annotations.is_suspicious),
            "stage2": holdout_metrics(index, stage2, annotations.is_suspicious)}


def fit_final(x, y, groups, context) -> dict:
    """Fit both stages on all labelled train rows (stage-2 context from nested OOF scores)."""
    all_rows = np.arange(len(y))
    stage1_model = build_model().fit(x, y)
    oof = nested_stage1(x, y, groups, all_rows, SEED + 21)
    full = pd.concat([x, context.aggregate(pd.Series(oof, index=x.index)).loc[x.index]], axis=1)
    stage2_model = build_model().fit(full, y)
    return {"stage1": stage1_model, "stage2": stage2_model, "features": list(x.columns),
            "stage2_features": list(full.columns)}


def predict_world(models: dict, x: pd.DataFrame, context: ScoreContext) -> np.ndarray:
    require(list(x.columns) == models["features"], "Feature columns differ from training")
    stage1 = models["stage1"].predict_proba(x)[:, 1]
    full = pd.concat([x, context.aggregate(pd.Series(stage1, index=x.index)).loc[x.index]], axis=1)
    require(list(full.columns) == models["stage2_features"], "Stage-2 columns differ from training")
    return models["stage2"].predict_proba(full)[:, 1]


def write_predictions(path: Path, index: pd.Index, scores: np.ndarray, expected: pd.Index) -> None:
    predictions = pd.DataFrame({"TRANSACTION_HASH": index, "score": scores})
    validate_predictions(predictions, expected)
    path.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(path, index=False, float_format="%.17g")
    reread = pd.read_csv(path, float_precision="round_trip")
    validate_predictions(reread, expected)
    require(reread.TRANSACTION_HASH.tolist() == list(index), "Serialized prediction order changed")
    require(np.array_equal(reread.score.to_numpy(), scores), "Scores changed during serialization")


EXTRA_MODULES: list[Path] = []  # optional add-on feature modules: compute(transactions) -> DataFrame


def extra_features(split: str | None, transactions: pd.DataFrame,
                   modules: list[Path] | None = None) -> list[pd.DataFrame]:
    """Label-free add-on features from each module, cached per split and module content hash.

    split=None computes without caching (used for the isolated stress-test subsets).
    """
    frames = []
    for module_path in (EXTRA_MODULES if modules is None else modules):
        digest = sha256(module_path)[:16]
        cache = output_path(f"extra_features/{module_path.stem}_{digest}/{split}.parquet") if split else None
        if cache is not None and cache.exists():
            frame = pd.read_parquet(cache).set_index("TRANSACTION_HASH")
        else:
            spec = importlib.util.spec_from_file_location(f"extra_{module_path.stem}", module_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            frame = module.compute(transactions.copy()).astype("float64")
            frame.index.name = "TRANSACTION_HASH"
            if cache is not None:
                cache.parent.mkdir(parents=True, exist_ok=True)
                frame.reset_index().to_parquet(cache, index=False)
        frame = frame.reindex(pd.Index(transactions.TRANSACTION_HASH))
        require(frame.index.is_unique and len(frame) == len(transactions), f"{module_path} row coverage")
        require(not np.isinf(frame.to_numpy()).any(), f"{module_path} infinite values")
        frames.append(frame.add_prefix(f"{module_path.stem}__"))
    return frames


def load_world(split: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    transactions = pd.read_parquet(output_path(f"{TABLE_RUN}/{split}/transactions.parquet"))
    features = pd.read_parquet(output_path(f"{FEATURE_RUN}/{split}/features.parquet")).set_index("TRANSACTION_HASH")
    require(features.index.equals(pd.Index(transactions.TRANSACTION_HASH)), f"{split} feature alignment")
    extras = extra_features(split, transactions)
    if extras:
        features = pd.concat([features, *extras], axis=1)
        require(features.columns.is_unique, "Duplicate feature names")
    return transactions, features


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(output_dir: Path, report_path: Path, validate: bool) -> dict:
    started = time.perf_counter()
    transactions, x = load_world("train")
    annotations = pd.read_csv(TRAIN / "labels.csv").set_index("TRANSACTION_HASH")
    require(annotations.index.is_unique and set(annotations.index) == set(x.index), "Label coverage")
    annotations = annotations.loc[x.index, list(ANNOTATIONS)]
    require(not set(ANNOTATIONS) & set(x.columns), "Annotation column among features")
    y = annotations.is_suspicious.to_numpy()
    groups = scenario_groups(annotations)
    context = ScoreContext(transactions)
    report: dict = {"run_id": output_dir.name, "seed": SEED, "model_parameters": MODEL_PARAMS,
                    "algorithm": "sklearn HistGradientBoostingClassifier, two stages",
                    "n_features_stage1": x.shape[1]}
    output_dir.mkdir(parents=True, exist_ok=True)
    with threadpool_limits(limits=THREADS):
        if validate:
            report["grouped_cv"], oof = grouped_cv(x, y, groups, context, annotations)
            print("grouped CV", json.dumps(report["grouped_cv"]["stage2"]), flush=True)
            write_predictions(output_dir / "train_oof_predictions.csv", x.index, oof, x.index)
            stress = Parallel(n_jobs=2)(
                delayed(_limited)(wallet_stress, transactions, x, y, groups, context, annotations, recompute,
                                  list(EXTRA_MODULES))
                for recompute in (False, True))
            for key, result in zip(("wallet_stress_full_week", "wallet_stress_isolated"), stress):
                report[key] = result
                print(key, json.dumps(result["stage2"]), flush=True)
        models = fit_final(x, y, groups, context)
        joblib.dump(models, output_dir / "models.joblib")
        test_transactions, x_test = load_world("test")
        test_scores = predict_world(models, x_test, ScoreContext(test_transactions))
    expected = pd.Index(pd.read_parquet(ROOT / "data/test_data/transactions.parquet",
                                        columns=["TRANSACTION_HASH"]).TRANSACTION_HASH)
    require(len(expected) == 40101, "Unexpected test transaction count")
    write_predictions(output_dir / "predictions.csv", x_test.index, test_scores, expected)
    report["test_predictions"] = {
        "rows": int(len(test_scores)), "distinct_scores": int(pd.Series(test_scores).nunique()),
        "min": float(test_scores.min()), "max": float(test_scores.max()),
        "mean": float(test_scores.mean()),
        "share_above_0.5": float((test_scores > 0.5).mean()),
        "sha256": sha256(output_dir / "predictions.csv")}
    report["versions"] = {p: importlib.metadata.version(p) for p in
                          ["numpy", "pandas", "pyarrow", "scikit-learn", "joblib", "threadpoolctl"]}
    report["python"] = platform.python_version()
    report["platform"] = platform.platform()
    report["seconds"] = round(time.perf_counter() - started, 1)
    sources = [Path(__file__), ROOT / "feature_workspace/aml_features.py",
               ROOT / "feature_workspace/score_context.py", TRAIN / "labels.csv"]
    report["source_sha256"] = {str(p.relative_to(ROOT)): sha256(p) for p in sources}
    report["extra_modules"] = {str(p): sha256(p) for p in EXTRA_MODULES}
    report["feature_run"] = FEATURE_RUN
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    (output_dir / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["test_predictions"], indent=2))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/aml_two_stage_v2")
    parser.add_argument("--report", type=Path, default=ROOT / "reports/aml_model_metrics_v2.json")
    parser.add_argument("--feature-run", default=FEATURE_RUN, help="base feature run under generated/")
    parser.add_argument("--extra-module", type=Path, action="append", default=[],
                        help="add-on feature module with compute(transactions); repeatable")
    parser.add_argument("--skip-validation", action="store_true",
                        help="Only fit the final model and write test predictions")
    args = parser.parse_args()
    FEATURE_RUN = args.feature_run
    EXTRA_MODULES[:] = [path.resolve() for path in args.extra_module]
    run(args.output_dir, args.report, validate=not args.skip_validation)
