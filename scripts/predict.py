"""
Simple CLI to run predictions on a CSV for classification or regression.

Examples:
- Classification: python scripts/predict.py --task classification --csv data/sample_unlabeled.csv
- Regression:     python scripts/predict.py --task regression --reg-target radius_mean --csv data/sample_unlabeled.csv
- Both (two outputs): python scripts/predict.py --task both --reg-target radius_mean --csv data/sample_unlabeled.csv

Requirements:
- Trained artifacts must exist in `artifacts/`:
  - `classifier_best.joblib`
  - `regressor_<target>_best.joblib`
"""

import argparse
import os
from pathlib import Path
from datetime import datetime

import pandas as pd
from joblib import load

from src.cancer_pipeline import load_dataset


REPORTS_DIR = Path("reports")
ARTIFACTS_DIR = Path("artifacts")


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Lowercase and replace spaces with underscores for column names."""
    df = df.copy()
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    return df


def ensure_reports_dir():
    REPORTS_DIR.mkdir(exist_ok=True)


def predict_classification(csv_path: str) -> str:
    """Run classification predictions using the saved best model.

    Returns path to the saved predictions CSV.
    """
    if not (ARTIFACTS_DIR / "classifier_best.joblib").exists():
        raise FileNotFoundError("Missing artifacts/classifier_best.joblib. Run training first.")

    # Template defines expected feature order
    df_template, _ = load_dataset()
    feature_cols = [c for c in df_template.columns if c != "target"]

    df_new = pd.read_csv(csv_path)
    df_new = normalize_columns(df_new)

    missing = [c for c in feature_cols if c not in df_new.columns]
    if missing:
        raise ValueError(f"CSV missing required columns for classification: {missing}")

    model = load(ARTIFACTS_DIR / "classifier_best.joblib")
    preds = model.predict(df_new[feature_cols])
    out = df_new.copy()
    out["prediction"] = preds

    ensure_reports_dir()
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_file = REPORTS_DIR / f"predictions_classification_{stamp}.csv"
    out.to_csv(out_file, index=False)
    print(f"Saved classification predictions to: {out_file}")
    return str(out_file)


def predict_regression(csv_path: str, target: str) -> str:
    """Run regression predictions for the given target using the saved best model.

    Returns path to the saved predictions CSV.
    """
    model_path = ARTIFACTS_DIR / f"regressor_{target}_best.joblib"
    if not model_path.exists():
        raise FileNotFoundError(f"Missing {model_path}. Run training first or choose a different target.")

    df_template, _ = load_dataset()
    feature_cols = [c for c in df_template.columns if c not in ("target", target)]

    df_new = pd.read_csv(csv_path)
    df_new = normalize_columns(df_new)

    missing = [c for c in feature_cols if c not in df_new.columns]
    if missing:
        raise ValueError(f"CSV missing required columns for regression[{target}]: {missing}")

    model = load(model_path)
    preds = model.predict(df_new[feature_cols])
    out = df_new.copy()
    out["prediction"] = preds

    ensure_reports_dir()
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_file = REPORTS_DIR / f"predictions_regression_{target}_{stamp}.csv"
    out.to_csv(out_file, index=False)
    print(f"Saved regression predictions to: {out_file}")
    return str(out_file)


def main():
    parser = argparse.ArgumentParser(description="Run predictions on a CSV using saved models.")
    parser.add_argument("--task", choices=["classification", "regression", "both"], default="classification")
    parser.add_argument("--csv", required=True, help="Path to input CSV containing feature columns")
    parser.add_argument("--reg-target", default="radius_mean", help="Regression target feature (default: radius_mean)")
    args = parser.parse_args()

    if args.task == "classification":
        predict_classification(args.csv)
    elif args.task == "regression":
        predict_regression(args.csv, args.reg_target)
    else:
        predict_classification(args.csv)
        predict_regression(args.csv, args.reg_target)


if __name__ == "__main__":
    main()

