"""
predictor.py — Model loading and inference logic.

Loads model.pkl once at startup (singleton pattern) and exposes
a single `predict()` function used by both API endpoints.
"""

from __future__ import annotations

import json
import warnings
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING

import joblib
import numpy as np

from src.utils import MODELS_DIR, PROJECT_ROOT, get_logger

if TYPE_CHECKING:
    from sklearn.ensemble import RandomForestClassifier

logger = get_logger(__name__)

# Feature order must match training exactly (see features.json)
FEATURE_ORDER = [
    "temperature_2m_max",
    "relative_humidity_2m_max",
    "wind_speed_10m_max",
    "rolling_max_temp_3d",
]

# Label classes in the same order as the classifier's classes_
LABEL_CLASSES = ["Low", "Moderate", "High"]


@lru_cache(maxsize=1)
def _load_model() -> "RandomForestClassifier":
    """Load and cache the trained RandomForestClassifier from model.pkl."""
    candidate_paths = [
        MODELS_DIR / "model.pkl",
        PROJECT_ROOT / "models" / "model.pkl",
        Path.cwd() / "models" / "model.pkl",
    ]
    model_path = next((p for p in candidate_paths if p.exists()), None)
    if model_path is None:
        raise FileNotFoundError(
            f"model.pkl not found in {[str(p) for p in candidate_paths]}. "
            "Run `python main.py` first to train and save the model."
        )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        clf = joblib.load(model_path)
    logger.info("Model loaded from %s", model_path)
    return clf


def predict(
    max_temp: float,
    humidity: float,
    wind_speed: float,
    rolling_max_temp_3d: float | None = None,
) -> dict:
    """
    Run inference and return risk level with confidence.

    Parameters
    ----------
    max_temp : float
        Daily maximum air temperature (°C).
    humidity : float
        Daily maximum relative humidity (%).
    wind_speed : float
        Daily maximum wind speed (km/h).
    rolling_max_temp_3d : float | None
        3-day rolling mean of max temperature (°C).
        If omitted, ``max_temp`` is used as a conservative fallback.

    Returns
    -------
    dict with keys:
        risk_level   : str   — "High" | "Moderate" | "Low"
        confidence   : float — probability of the predicted class [0, 1]
        probabilities: dict  — per-class probabilities
        features_used: dict  — the exact feature vector sent to the model
    """
    if rolling_max_temp_3d is None:
        # Fallback: assume temperature hasn't changed much in the last 3 days
        rolling_max_temp_3d = max_temp
        logger.debug(
            "rolling_max_temp_3d not provided; using max_temp=%.1f as fallback",
            max_temp,
        )

    features = np.array(
        [[max_temp, humidity, wind_speed, rolling_max_temp_3d]], dtype=float
    )

    clf = _load_model()

    # Align class labels with the model's internal class ordering
    model_classes: list[str] = list(clf.classes_)
    proba_raw: np.ndarray = clf.predict_proba(features)[0]

    # Build a label → probability dict aligned to the model's class order
    proba_dict: dict[str, float] = {
        cls: round(float(p), 4) for cls, p in zip(model_classes, proba_raw)
    }

    # Predicted class and its confidence
    predicted_label: str = clf.predict(features)[0]
    confidence: float = round(float(proba_dict[predicted_label]), 4)

    return {
        "risk_level": predicted_label,
        "confidence": confidence,
        "probabilities": {lbl: proba_dict.get(lbl, 0.0) for lbl in LABEL_CLASSES},
        "features_used": {
            "temperature_2m_max": round(max_temp, 2),
            "relative_humidity_2m_max": round(humidity, 2),
            "wind_speed_10m_max": round(wind_speed, 2),
            "rolling_max_temp_3d": round(rolling_max_temp_3d, 2),
        },
    }
