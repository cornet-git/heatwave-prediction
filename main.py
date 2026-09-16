"""
main.py — Orchestrator for the Heatwave Risk Prediction pipeline.

Pipeline stages
---------------
  1. Fetch  — pull 5 years of daily weather from Open-Meteo for Mumbai & Pune
  2. Save   — persist raw data to data/weather_raw.csv
  3. Engineer — compute rolling features + assign risk labels
  4. Save   — persist processed data to data/weather_processed.csv
  5. Train  — fit Random Forest, evaluate, export model.pkl + features.json

Run
---
    python main.py
"""

import sys
from pathlib import Path

# Ensure the project root is on sys.path so `src` is importable regardless
# of how Python was launched (standard install, virtual env, embedded, etc.)
_PROJECT_ROOT = Path(__file__).resolve().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.data_fetch import fetch_all_cities
from src.feature_engineering import engineer_features
from src.train import train_and_evaluate
from src.utils import DATA_DIR, get_logger

logger = get_logger("main")


def main() -> None:
    logger.info("╔══════════════════════════════════════════════════╗")
    logger.info("║  Climate Intelligence — Heatwave Risk Prediction  ║")
    logger.info("╚══════════════════════════════════════════════════╝")

    # ------------------------------------------------------------------
    # Stage 1: Data Fetch
    # ------------------------------------------------------------------
    logger.info("STAGE 1 — Fetching historical weather data …")
    try:
        raw_df = fetch_all_cities(years=5)
    except RuntimeError as exc:
        logger.error("Data fetch failed: %s", exc)
        sys.exit(1)

    raw_path = DATA_DIR / "weather_raw.csv"
    raw_df.to_csv(raw_path, index=False)
    logger.info("Raw data saved → %s  (%d rows)", raw_path, len(raw_df))

    # ------------------------------------------------------------------
    # Stage 2: Feature Engineering
    # ------------------------------------------------------------------
    logger.info("STAGE 2 — Engineering features and labelling …")
    processed_df = engineer_features(raw_df)

    processed_path = DATA_DIR / "weather_processed.csv"
    processed_df.to_csv(processed_path, index=False)
    logger.info(
        "Processed data saved → %s  (%d rows)", processed_path, len(processed_df)
    )

    # ------------------------------------------------------------------
    # Stage 3: Model Training & Evaluation
    # ------------------------------------------------------------------
    logger.info("STAGE 3 — Training heatwave risk classifier …")
    train_and_evaluate(processed_df)

    logger.info("Pipeline complete. Artefacts written to models/ directory.")


if __name__ == "__main__":
    main()
