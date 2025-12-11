"""
Validation script for saved models.

This utility loads the breast cancer dataset, evaluates predictions using
the saved classifier and regressor artifacts, and writes:
- `reports/validation.json` with summary metrics.
- `reports/validation_classification_samples.csv` with example true vs pred.
- `reports/validation_regression_samples.csv` with example true vs pred.

Run it after training to produce validation artifacts used by the notebook.
"""

import os
import json
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import load
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer

ARTIFACTS_DIR = "artifacts"
REPORTS_DIR = "reports"


def load_dataset():
    """Load and rename the breast cancer dataset to match pipeline naming."""
    data = load_breast_cancer()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df["target"] = data.target
    renamed = {
        "mean radius": "radius_mean",
        "mean texture": "texture_mean",
        "mean perimeter": "perimeter_mean",
        "mean area": "area_mean",
        "mean smoothness": "smoothness_mean",
        "mean compactness": "compactness_mean",
        "mean concavity": "concavity_mean",
        "mean concave points": "concave_points_mean",
        "mean symmetry": "symmetry_mean",
        "mean fractal dimension": "fractal_dimension_mean",
        "radius error": "radius_se",
        "texture error": "texture_se",
        "perimeter error": "perimeter_se",
        "area error": "area_se",
        "smoothness error": "smoothness_se",
        "compactness error": "compactness_se",
        "concavity error": "concavity_se",
        "concave points error": "concave_points_se",
        "symmetry error": "symmetry_se",
        "fractal dimension error": "fractal_dimension_se",
        "worst radius": "radius_worst",
        "worst texture": "texture_worst",
        "worst perimeter": "perimeter_worst",
        "worst area": "area_worst",
        "worst smoothness": "smoothness_worst",
        "worst compactness": "compactness_worst",
        "worst concavity": "concavity_worst",
        "worst concave points": "concave_points_worst",
        "worst symmetry": "symmetry_worst",
        "worst fractal dimension": "fractal_dimension_worst",
    }
    df = df.rename(columns=renamed)
    return df


def validate_classification(df: pd.DataFrame):
    """Evaluate the saved classifier on a holdout split.

    Returns a tuple of (metrics_dict, sample_dataframe) where the sample contains
    a small set of true vs predicted labels.
    """
    model_path = os.path.join(ARTIFACTS_DIR, "classifier_best.joblib")
    if not os.path.exists(model_path):
        raise FileNotFoundError("Classifier artifact not found. Train models first.")

    X = df.drop(columns=["target"])
    y = df["target"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = load(model_path)
    y_pred = model.predict(X_test)

    # try to get probabilities or decision scores for ROC-AUC
    try:
        if hasattr(model, "predict_proba"):
            y_score = model.predict_proba(X_test)[:, 1]
        else:
            y_score = model.decision_function(X_test)
            y_score = (y_score - y_score.min()) / (y_score.max() - y_score.min())
    except Exception:
        y_score = None

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred)),
        "recall": float(recall_score(y_test, y_pred)),
        "f1": float(f1_score(y_test, y_pred)),
        "roc_auc": float(roc_auc_score(y_test, y_score)) if y_score is not None else None,
    }

    # sample comparisons
    sample_n = min(10, len(X_test))
    sample = pd.DataFrame({
        "true": y_test.iloc[:sample_n].values,
        "pred": y_pred[:sample_n],
    })

    return metrics, sample


def validate_regression(df: pd.DataFrame, target: str = "radius_mean"):
    """Evaluate the saved regressor for a given target on a holdout split.

    Returns a tuple of (metrics_dict, sample_dataframe) where the sample contains
    a small set of true vs predicted values.
    """
    model_path = os.path.join(ARTIFACTS_DIR, f"regressor_{target}_best.joblib")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Regressor artifact for target '{target}' not found. Train models first.")

    feature_cols = [c for c in df.columns if c not in ("target", target)]
    X = df[feature_cols]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = load(model_path)
    y_pred = model.predict(X_test)

    metrics = {
        "MAE": float(mean_absolute_error(y_test, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_test, y_pred))),
        "R2": float(r2_score(y_test, y_pred)),
    }

    sample_n = min(10, len(X_test))
    sample = pd.DataFrame({
        "true": y_test.iloc[:sample_n].values,
        "pred": y_pred[:sample_n],
    })

    return metrics, sample


def main():
    """Produce validation metrics and sample comparisons for both tasks."""
    Path(REPORTS_DIR).mkdir(exist_ok=True)
    df = load_dataset()

    cls_metrics, cls_sample = validate_classification(df)
    reg_metrics, reg_sample = validate_regression(df, target="radius_mean")

    # Save validation outputs
    out = {
        "classification": cls_metrics,
        "regression_radius_mean": reg_metrics,
    }
    with open(os.path.join(REPORTS_DIR, "validation.json"), "w") as f:
        json.dump(out, f, indent=2)

    cls_sample.to_csv(os.path.join(REPORTS_DIR, "validation_classification_samples.csv"), index=False)
    reg_sample.to_csv(os.path.join(REPORTS_DIR, "validation_regression_samples.csv"), index=False)

    print("Validation complete. Results saved to reports/validation.json and sample comparisons CSVs.")


if __name__ == "__main__":
    main()
