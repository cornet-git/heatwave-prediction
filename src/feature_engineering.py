"""
feature_engineering.py — Feature engineering and heatwave risk labelling.

Features produced
-----------------
  - temperature_2m_max           (°C)        — raw max temperature
  - relative_humidity_2m_max     (%)         — raw max humidity
  - wind_speed_10m_max           (km/h)      — raw max wind speed
  - rolling_max_temp_3d          (°C)        — 3-day rolling mean of max temp

Labelling rules (applied in priority order)
--------------------------------------------
  HIGH     : max_temp >= 40 °C  OR  (max_temp - rolling_avg) >= 4.5 °C
  MODERATE : 35 °C <= max_temp < 40 °C
  LOW      : max_temp < 35 °C

The rolling average is computed per city so that no city's data
contaminates another city's rolling window.
"""

import numpy as np
import pandas as pd

from src.utils import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
ROLLING_WINDOW       = 3      # days
HIGH_TEMP_THRESHOLD  = 40.0   # °C  — absolute high
MOD_TEMP_THRESHOLD   = 35.0   # °C  — moderate lower bound
SPIKE_THRESHOLD      = 4.5    # °C  — deviation above rolling avg → High

FEATURE_COLUMNS = [
    "temperature_2m_max",
    "relative_humidity_2m_max",
    "wind_speed_10m_max",
    "rolling_max_temp_3d",
]

LABEL_COLUMN = "risk_label"


def _compute_rolling(group: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the 3-day rolling mean of max temperature within a city group.
    Uses a closed-left window (past 3 days, not including current day) to
    avoid data leakage.
    """
    group = group.sort_values("date").copy()
    # min_periods=1 so we don't lose the first two days entirely,
    # but we'll drop NaN rows caused by insufficient window later.
    group["rolling_max_temp_3d"] = (
        group["temperature_2m_max"]
        .rolling(window=ROLLING_WINDOW, min_periods=ROLLING_WINDOW)
        .mean()
    )
    return group


def _assign_label(row: pd.Series) -> str:
    """Assign a single risk label to one observation."""
    max_temp = row["temperature_2m_max"]
    rolling  = row["rolling_max_temp_3d"]
    deviation = max_temp - rolling

    if max_temp >= HIGH_TEMP_THRESHOLD or deviation >= SPIKE_THRESHOLD:
        return "High"
    if max_temp >= MOD_TEMP_THRESHOLD:
        return "Moderate"
    return "Low"


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add engineered features and risk labels to the raw weather DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Raw weather data as returned by `data_fetch.fetch_all_cities`.

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame with feature columns and ``risk_label`` column.
        Rows with NaN feature values (first 2 days per city due to rolling
        window) are dropped.
    """
    logger.info("Engineering features …")

    # Compute rolling average per city (keeps city windows separate)
    df = (
        df.groupby("city", group_keys=False)
        .apply(_compute_rolling)
        .reset_index(drop=True)
    )

    # Drop rows where rolling average is NaN (insufficient history)
    before = len(df)
    df.dropna(subset=["rolling_max_temp_3d"], inplace=True)
    dropped = before - len(df)
    if dropped:
        logger.info("  Dropped %d rows with incomplete rolling window.", dropped)

    # Assign risk labels
    df[LABEL_COLUMN] = df.apply(_assign_label, axis=1)

    # Log class distribution
    dist = df[LABEL_COLUMN].value_counts()
    logger.info("Risk label distribution:\n%s", dist.to_string())

    logger.info("Feature engineering complete. Dataset shape: %s", df.shape)
    return df.reset_index(drop=True)
