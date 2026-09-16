"""
train.py — Trains a Random Forest classifier to predict heatwave risk level,
evaluates it, and persists the trained artefacts.

Outputs
-------
  models/model.pkl       — serialised RandomForestClassifier (via joblib)
  models/features.json   — expected input schema for inference
"""

import json

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split

from src.feature_engineering import FEATURE_COLUMNS, LABEL_COLUMN
from src.utils import MODELS_DIR, get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Hyper-parameters
# ---------------------------------------------------------------------------
N_ESTIMATORS  = 200
RANDOM_STATE  = 42
TEST_SIZE     = 0.20   # 80/20 split

# Label order for consistent display
LABEL_ORDER = ["Low", "Moderate", "High"]


def train_and_evaluate(df: pd.DataFrame) -> RandomForestClassifier:
    """
    Train a Random Forest on the engineered dataset, print evaluation metrics,
    and save model artefacts.

    Parameters
    ----------
    df : pd.DataFrame
        Feature-engineered DataFrame produced by `feature_engineering.engineer_features`.

    Returns
    -------
    RandomForestClassifier
        The trained classifier (also saved to disk).
    """
    logger.info("Preparing training data …")

    X = df[FEATURE_COLUMNS].values
    y = df[LABEL_COLUMN].values

    # Stratified split preserves class ratios in both sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    logger.info(
        "Split — train: %d samples | test: %d samples", len(X_train), len(X_test)
    )

    # ------------------------------------------------------------------ train
    logger.info(
        "Training RandomForestClassifier (n_estimators=%d, class_weight='balanced') …",
        N_ESTIMATORS,
    )
    clf = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        class_weight="balanced",     # compensates for class imbalance
        random_state=RANDOM_STATE,
        n_jobs=-1,                   # use all available CPU cores
    )
    clf.fit(X_train, y_train)
    logger.info("Training complete.")

    # --------------------------------------------------------------- evaluate
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    # Pretty-print evaluation results
    separator = "=" * 60
    print(f"\n{separator}")
    print("  HEATWAVE RISK PREDICTION — MODEL EVALUATION")
    print(separator)
    print(f"  Overall Accuracy : {accuracy * 100:.2f}%")
    print(separator)
    print("\n  Per-Class Report (Precision / Recall / F1-Score):\n")
    report = classification_report(
        y_test, y_pred,
        labels=LABEL_ORDER,
        target_names=LABEL_ORDER,
        digits=4,
        zero_division=0,
    )
    print(report)
    print(separator + "\n")

    # Feature importances
    importances = clf.feature_importances_
    print("  Feature Importances:")
    for feat, imp in sorted(
        zip(FEATURE_COLUMNS, importances), key=lambda x: x[1], reverse=True
    ):
        print(f"    {feat:<35} {imp:.4f}")
    print()

    # ------------------------------------------------------------------ save
    _save_artefacts(clf, df)

    return clf


def _save_artefacts(clf: RandomForestClassifier, df: pd.DataFrame) -> None:
    """Persist model.pkl and features.json to the models/ directory."""

    # 1. model.pkl
    model_path = MODELS_DIR / "model.pkl"
    joblib.dump(clf, model_path)
    logger.info("Model saved → %s", model_path)

    # 2. features.json — describes the expected inference schema
    feature_schema = {
        "description": (
            "Input schema for the heatwave risk prediction model. "
            "Supply one record per day per city."
        ),
        "model_type": "RandomForestClassifier",
        "label_classes": LABEL_ORDER,
        "features": [
            {
                "name": "temperature_2m_max",
                "type": "float",
                "unit": "°C",
                "description": "Maximum air temperature at 2 m above ground.",
            },
            {
                "name": "relative_humidity_2m_max",
                "type": "float",
                "unit": "%",
                "description": "Maximum relative humidity at 2 m above ground.",
            },
            {
                "name": "wind_speed_10m_max",
                "type": "float",
                "unit": "km/h",
                "description": "Maximum wind speed at 10 m above ground.",
            },
            {
                "name": "rolling_max_temp_3d",
                "type": "float",
                "unit": "°C",
                "description": (
                    "3-day rolling mean of temperature_2m_max "
                    "(computed per city; requires at least 3 prior days)."
                ),
            },
        ],
        "training_stats": {
            "n_estimators": N_ESTIMATORS,
            "test_size_fraction": TEST_SIZE,
            "class_weight": "balanced",
            "feature_order": FEATURE_COLUMNS,
            "train_samples": int(df.shape[0] * (1 - TEST_SIZE)),
            "label_distribution": df[LABEL_COLUMN].value_counts().to_dict(),
        },
    }

    features_path = MODELS_DIR / "features.json"
    with open(features_path, "w", encoding="utf-8") as fh:
        json.dump(feature_schema, fh, indent=2)
    logger.info("Feature schema saved → %s", features_path)
