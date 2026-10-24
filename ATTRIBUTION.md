# What we did not write ourselves

## Organiser materials
- `data/traning_data/payments.py` and `starter_notebook.ipynb` (supplied with the training data, not included in this ZIP). `scripts/baseline.py` adapts the notebook's twelve starter features and logistic regression; the preparation step cross-checks its USD values against `payments.py`.
- Datasets (train, practice validation, test) are the organisers' and are not included.

## Libraries (versions pinned in `requirements-baseline.txt`, Python 3.10.15)
- NumPy, pandas, PyArrow, SciPy, scikit-learn (`HistGradientBoostingClassifier`, `StratifiedGroupKFold`, `LogisticRegression`, metrics), joblib, threadpoolctl, plus their transitive dependencies (python-dateutil, pytz, tzdata, six).
- LightGBM 4.6.0 was used only in a one-off model comparison during development; the final pipeline does not use it.

## AI assistance
- OpenAI Codex (GPT-6) helped implement and analyse the data audit, payment preparation, wallet graphs and starter baseline.
- Anthropic Claude Code (model Claude Opus 5.5, `claude-opus-5-5`) helped implement and analyse the AML features (`feature_workspace/aml_features.py`), the score-context stage (`feature_workspace/score_context.py`), the two-stage model and validation (`scripts/aml_model.py`) and the tests (`tests/test_aml_features.py`). It was prompted to build and validate AML features and a model that maximises held-out Average Precision. All analytical behaviour is defined by the executable code here; no LLM is called at run time.

## Not used
- No external transaction data, outside services, pretrained models, scraped addresses or hidden labels.
