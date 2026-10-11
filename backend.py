
"""
Backend for the Water Quality Predictor (SDG 6: Clean Water and Sanitation).

Everything related to data, machine learning and prediction lives here.
The Streamlit frontend (app.py) only calls the functions in this file.

You can also run this file directly to train and save the model:
    python backend.py
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_CSV = BASE_DIR / "water_potability.csv"
MODEL_PATH = BASE_DIR / "model.joblib"

TARGET = "Potability"
FEATURES = [
    "ph",
    "Hardness",
    "Solids",
    "Chloramines",
    "Sulfate",
    "Conductivity",
    "Organic_carbon",
    "Trihalomethanes",
    "Turbidity",
]

# Label, unit and a short explanation shown in the app for each feature.
FEATURE_INFO = {
    "ph": ("pH", "", "Acid-base balance of the water."),
    "Hardness": ("Hardness", "mg/L", "Calcium and magnesium dissolved in the water."),
    "Solids": ("Total dissolved solids", "ppm", "All minerals and salts dissolved in the water."),
    "Chloramines": ("Chloramines", "ppm", "Disinfectant used in public water systems."),
    "Sulfate": ("Sulfate", "mg/L", "Naturally occurring substance found in water and soil."),
    "Conductivity": ("Conductivity", "µS/cm", "How well the water conducts electricity (saltiness)."),
    "Organic_carbon": ("Organic carbon", "ppm", "Amount of carbon from organic compounds."),
    "Trihalomethanes": ("Trihalomethanes", "µg/L", "By-products formed when chlorine treats water."),
    "Turbidity": ("Turbidity", "NTU", "Cloudiness of the water caused by suspended particles."),
}


@dataclass
class ModelBundle:
    """The trained model together with everything the app needs to display."""

    pipeline: object
    metrics: dict
    feature_stats: dict  # per feature: min, max, median (from the training data)
    data_summary: dict   # rows, class counts, missing values


# ---------------------------------------------------------------
# Data loading and validation
# ---------------------------------------------------------------
def load_dataset(source) -> pd.DataFrame:
    """Read the CSV (a path or an uploaded file) and check that it is usable."""
    df = pd.read_csv(source)

    missing_cols = [c for c in FEATURES + [TARGET] if c not in df.columns]
    if missing_cols:
        raise ValueError(
            "The dataset is missing these columns: " + ", ".join(missing_cols)
        )

    df = df[FEATURES + [TARGET]]
    try:
        df = df.apply(pd.to_numeric, errors="raise")
    except (TypeError, ValueError) as error:
        raise ValueError("All columns must contain numeric values.") from error

    if df[TARGET].isna().any():
        raise ValueError(f"The '{TARGET}' column contains missing values.")
    if set(df[TARGET].unique()) - {0, 1}:
        raise ValueError(f"The '{TARGET}' column must only contain 0 and 1.")
    return df


# ---------------------------------------------------------------
# Training
# ---------------------------------------------------------------
def train(df: pd.DataFrame) -> ModelBundle:
    """Train a Random Forest and evaluate it on a held-out test set."""
    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # The imputer is fitted on training data only, so there is no data leakage.
    pipeline = make_pipeline(
        SimpleImputer(strategy="median"),
        RandomForestClassifier(
            n_estimators=200, random_state=42, class_weight="balanced"
        ),
    )
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    report = classification_report(
        y_test, y_pred, labels=[0, 1], output_dict=True, zero_division=0
    )
    forest = pipeline.named_steps["randomforestclassifier"]

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "report": report,
        "confusion_matrix": confusion_matrix(y_test, y_pred, labels=[0, 1]).tolist(),
        "importances": dict(zip(FEATURES, map(float, forest.feature_importances_))),
        "train_size": int(len(X_train)),
        "test_size": int(len(X_test)),
    }
    feature_stats = {
        f: {
            "min": float(X[f].min()),
            "max": float(X[f].max()),
            "median": float(X[f].median()),
        }
        for f in FEATURES
    }
    data_summary = {
        "rows": int(len(df)),
        "class_counts": {int(k): int(v) for k, v in y.value_counts().items()},
        "missing": {f: int(X[f].isna().sum()) for f in FEATURES},
    }
    return ModelBundle(pipeline, metrics, feature_stats, data_summary)


# ---------------------------------------------------------------
# Saving and loading
# ---------------------------------------------------------------
def save_bundle(bundle: ModelBundle, path: Path = MODEL_PATH) -> None:
    joblib.dump(bundle, path)


def load_bundle(path: Path = MODEL_PATH) -> ModelBundle:
    return joblib.load(path)


# ---------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------
def predict(bundle: ModelBundle, values: dict, threshold: float = 0.5):
    """Return (is_potable, probability_of_potable) for one water sample."""
    missing = [f for f in FEATURES if f not in values]
    if missing:
        raise ValueError("Missing values for: " + ", ".join(missing))

    row = pd.DataFrame([values])[FEATURES]
    probabilities = bundle.pipeline.predict_proba(row)[0]
    potable_index = list(bundle.pipeline.classes_).index(1)
    probability = float(probabilities[potable_index])
    return probability >= threshold, probability


if __name__ == "__main__":
    # Import this file as a normal module so the saved model can be loaded by
    # app.py later (a class defined in __main__ cannot be unpickled elsewhere).
    import backend as be

    data = be.load_dataset(be.DEFAULT_CSV)
    result = be.train(data)
    be.save_bundle(result)
    print(f"Trained on {result.metrics['train_size']} rows, "
          f"tested on {result.metrics['test_size']} rows.")
    print("Accuracy:", round(result.metrics["accuracy"], 3))
    print("Model saved to:", be.MODEL_PATH)