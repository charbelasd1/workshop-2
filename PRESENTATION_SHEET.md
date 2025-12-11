Presentation Sheet: Who Explains What (All Files)

Roles
- You (Classification): Own classifier training, metrics, CLI predictions.
- Friend (Streamlit/UI & Design): Own app.py, UX, sample CSV integration, downloads, captions.
- Teammate 3 (Notebook Explainer): Own notebook story and executed HTML.
- Teammate 4 (Regression & Tooling): Own regressor, dataset exports, validation script, requirements.

Top-Level Files
- README.md — Presenter: Streamlit/UI
  - What to say: How to run app, pipeline, CLI; examples; where outputs go.
  - Demo cue: Scroll to prediction examples; match commands to live outputs.
- requirements.txt — Presenter: Regression & Tooling
  - What to say: Key libs (streamlit, sklearn, pandas, joblib); ensure venv installed.
- app.py — Presenter: Streamlit/UI
  - What to say: Data input (sample/upload), classification/regression buttons, manual input, class‑ID toggle, downloads, Model Report.
  - Demo cue: Predict table, flip class‑ID mapping, manual form edit to flip label.
- Breast_Cancer_Visualization_Preprocessing.ipynb — Presenter: Notebook Explainer
  - What to say: Dataset, features (mean/se/worst), scaling, quick visuals; how it connects to reports.
- Breast_Cancer_Visualization_Preprocessing_executed.html — Presenter: Notebook Explainer
  - What to say: Executed version for the audience; open and point to metrics section.
- data.csv — Presenter: Regression & Tooling
  - What to say: Full dataset export used by notebooks; standardized header names.
- workshop-2_submission.zip — Presenter: Regression & Tooling
  - What to say: Packaged submission; contains notebook, HTML, README.

Source Code
- src/cancer_pipeline.py — Presenters: Classification + Regression
  - What to say (Classification): Data load/rename, model training, ROC-AUC selection, artifacts saved to `artifacts/classifier_best.joblib`, figures/metrics to `reports/`.
  - What to say (Regression): Target `radius_mean`, features exclude target, metrics and artifact `artifacts/regressor_radius_mean_best.joblib`.
  - Demo cue: Run `python src/cancer_pipeline.py --task both` if artifacts missing.

Scripts
- scripts/predict.py — Presenter: Classification (primary), Regression (secondary)
  - What to say: Headless predictions for CSV; normalizes headers; saves to `reports/predictions_*.csv`.
  - Commands: Classification `python scripts/predict.py --task classification --csv data/sample_unlabeled.csv`; Regression `--task regression --reg-target radius_mean`.
- scripts/validate_predictions.py — Presenters: Classification + Regression
  - What to say: Holdout evaluation; writes `reports/validation.json` and sample comparisons CSVs.
  - Demo cue: Show numbers the notebook/UI use.
- scripts/export_sample_csv.py — Presenter: Streamlit/UI
  - What to say: Balanced sample (8 benign, 7 malignant) with feature-only columns; drives Data Input table.
  - Command: `python scripts/export_sample_csv.py`.
- scripts/export_full_dataset.py — Presenter: Regression & Tooling
  - What to say: Full dataset with `target`; standardized headers.
  - Command: `python scripts/export_full_dataset.py`.
- scripts/export_kaggle_like_dataset.py — Presenter: Regression & Tooling
  - What to say: Kaggle-style CSV with `diagnosis` ('M'/'B'), `id`, `Unnamed: 32` plus `target`.
  - Command: `python scripts/export_kaggle_like_dataset.py`.
- scripts/patch_notebook_add_metrics.py — Presenter: Notebook Explainer
  - What to say: Ensures notebook has metrics cells; reads validation CSVs.
  - Command: `python scripts/patch_notebook_add_metrics.py`.

Data Folder
- data/sample_unlabeled.csv — Presenter: Streamlit/UI
  - What to say: Balanced, shuffled demo input; auto-loaded; underscores in headers.

Generated During Runs
- artifacts/ — Presenters: Classification + Regression
  - What to say: Saved models (`classifier_best.joblib`, `regressor_radius_mean_best.joblib`).
- reports/ — Presenters: All
  - What to say: `metrics.json`, `validation.json`, `predictions_*.csv`, and `figures/` (confusion matrix, ROC, residuals, feature importances).

One-Page Demo Script
- Load sample in app → predict classification → show label + probabilities.
- Toggle class‑ID mapping → point out numeric `prediction` changes.
- Manual input → reduce worst features to flip label to Benign.
- Predict regression on same row → compare to `radius_mean` size.
- Open Model Report → tie metrics/figures to validation outputs.

Ownership Summary
- You: Classification and CLI; explain classifier parts in pipeline and validate metrics.
- Friend: Streamlit/UI and design; explain app.py and sample export; live demo driver.
- Teammate 3: Notebook; explain data/feature story and executed HTML.
- Teammate 4: Regression & tooling; explain regressor, exports, validation script, requirements.
