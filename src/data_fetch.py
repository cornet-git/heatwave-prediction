"""
data_fetch.py — Fetches historical daily weather data from the Open-Meteo
Historical Archive API (no API key required) for Mumbai and Pune.

API endpoint: https://archive-api.open-meteo.com/v1/archive
Docs: https://open-meteo.com/en/docs/historical-weather-api

Variables fetched (daily):
  - temperature_2m_max       : Maximum air temperature at 2 m (°C)
  - temperature_2m_min       : Minimum air temperature at 2 m (°C)
  - relative_humidity_2m_max : Maximum relative humidity at 2 m (%)
  - wind_speed_10m_max       : Maximum wind speed at 10 m (km/h)
"""

import time
from datetime import date, timedelta
from typing import Any

import pandas as pd
import requests

from src.utils import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# City registry
# ---------------------------------------------------------------------------
CITIES: dict[str, dict[str, float]] = {
    "Mumbai": {"latitude": 19.076,  "longitude": 72.877},
    "Pune":   {"latitude": 18.5204, "longitude": 73.8567},
}

# Open-Meteo archive endpoint
BASE_URL = "https://archive-api.open-meteo.com/v1/archive"

# Daily variables to request
DAILY_VARIABLES = [
    "temperature_2m_max",
    "temperature_2m_min",
    "relative_humidity_2m_max",
    "wind_speed_10m_max",
]


def _date_range(years: int = 5) -> tuple[str, str]:
    """Return (start_date, end_date) strings covering the last `years` years."""
    end   = date.today() - timedelta(days=1)   # yesterday (latest complete day)
    start = end.replace(year=end.year - years)
    return start.isoformat(), end.isoformat()


def _fetch_city(city: str, lat: float, lon: float,
                start: str, end: str,
                retries: int = 3) -> pd.DataFrame:
    """
    Fetch daily weather for one city from the Open-Meteo archive API.

    Returns a DataFrame with columns:
        date, city, temperature_2m_max, temperature_2m_min,
        relative_humidity_2m_max, wind_speed_10m_max
    """
    params: dict[str, Any] = {
        "latitude":  lat,
        "longitude": lon,
        "start_date": start,
        "end_date":   end,
        "daily":      ",".join(DAILY_VARIABLES),
        "timezone":   "Asia/Kolkata",
    }

    for attempt in range(1, retries + 1):
        try:
            logger.info(
                "Fetching %s data %s → %s (attempt %d/%d)",
                city, start, end, attempt, retries,
            )
            resp = requests.get(BASE_URL, params=params, timeout=60)
            resp.raise_for_status()
            payload = resp.json()

            daily = payload["daily"]
            df = pd.DataFrame({"date": daily["time"]})
            df["city"] = city
            for var in DAILY_VARIABLES:
                df[var] = daily[var]

            df["date"] = pd.to_datetime(df["date"])
            logger.info("  ✓ %s: %d records retrieved", city, len(df))
            return df

        except requests.RequestException as exc:
            logger.warning("  ✗ Attempt %d failed for %s: %s", attempt, city, exc)
            if attempt < retries:
                wait = 2 ** attempt        # exponential back-off
                logger.info("  Retrying in %ds …", wait)
                time.sleep(wait)
            else:
                raise RuntimeError(
                    f"Failed to fetch data for {city} after {retries} attempts."
                ) from exc


def fetch_all_cities(years: int = 5) -> pd.DataFrame:
    """
    Fetch historical weather for all cities and return a combined DataFrame.

    Parameters
    ----------
    years : int
        Number of historical years to fetch (default: 5).

    Returns
    -------
    pd.DataFrame
        Combined DataFrame sorted by city and date, with columns:
        date, city, temperature_2m_max, temperature_2m_min,
        relative_humidity_2m_max, wind_speed_10m_max
    """
    start, end = _date_range(years)
    logger.info("Date range: %s → %s", start, end)

    frames: list[pd.DataFrame] = []
    for city, coords in CITIES.items():
        df = _fetch_city(city, coords["latitude"], coords["longitude"], start, end)
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)
    combined.sort_values(["city", "date"], inplace=True)
    combined.reset_index(drop=True, inplace=True)

    logger.info("Total records fetched: %d", len(combined))
    return combined
