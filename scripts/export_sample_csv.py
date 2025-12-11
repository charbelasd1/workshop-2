"""
Export a small unlabeled sample CSV used by the Streamlit app.

Creates `data/sample_unlabeled.csv` with a roughly balanced mix of benign
and malignant rows selected from the scikit‑learn breast cancer dataset.
The output has ONLY feature columns (no `target`), suitable for predictions.
"""

from pathlib import Path
import pandas as pd
from sklearn.datasets import load_breast_cancer

# Load dataset and standardize column names like in the pipeline
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

data = load_breast_cancer()
features_df = pd.DataFrame(data.data, columns=data.feature_names).rename(columns=renamed)
labels = pd.Series(data.target, name="target")  # 0=malignant, 1=benign

# Build a balanced sample: 8 benign + 7 malignant (total 15), shuffled.
benign = features_df[labels == 1].sample(n=8, random_state=42)
malignant = features_df[labels == 0].sample(n=7, random_state=42)
sample = pd.concat([benign, malignant], axis=0).sample(frac=1.0, random_state=42).reset_index(drop=True)

Path("data").mkdir(parents=True, exist_ok=True)
sample.to_csv("data/sample_unlabeled.csv", index=False)
print(
    "Saved: data/sample_unlabeled.csv with",
    sample.shape[0], "rows and", sample.shape[1], "features (8 benign, 7 malignant)"
)
