"""
weather.py — Open-Meteo API clients for geocoding and forecast data.

Geocoding API   : https://geocoding-api.open-meteo.com/v1/search   (no key)
Forecast API    : https://api.open-meteo.com/v1/forecast            (no key)

The forecast endpoint is used to:
  - Retrieve today's weather for the /current-weather route
  - Retrieve the last 3 days of max temps to compute rolling_max_temp_3d
"""

from __future__ import annotations

import time
from datetime import date

import requests

from src.utils import get_logger

logger = get_logger(__name__)

GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_URL  = "https://api.open-meteo.com/v1/forecast"

DAILY_VARIABLES = [
    "temperature_2m_max",
    "temperature_2m_min",
    "relative_humidity_2m_max",
    "wind_speed_10m_max",
]

_REQUEST_TIMEOUT = 15  # seconds


# ---------------------------------------------------------------------------
# Geocoding
# ---------------------------------------------------------------------------

class CityNotFoundError(ValueError):
    """Raised when the geocoding API returns no results for a city name."""


def geocode_city(city: str) -> dict:
    """
    Resolve a city name to coordinates via the Open-Meteo Geocoding API.

    Parameters
    ----------
    city : str
        Human-readable city name (e.g. "Mumbai", "Delhi").

    Returns
    -------
    dict with keys: name, latitude, longitude, country, timezone

    Raises
    ------
    CityNotFoundError
        If the city cannot be resolved to coordinates.
    requests.RequestException
        On network failures.
    """
    params = {
        "name":     city.strip(),
        "count":    1,
        "language": "en",
        "format":   "json",
    }
    logger.debug("Geocoding city: %s", city)
    resp = requests.get(GEOCODING_URL, params=params, timeout=_REQUEST_TIMEOUT)
    resp.raise_for_status()
    data = resp.json()

    results = data.get("results")
    if not results:
        raise CityNotFoundError(
            f"City '{city}' not found. "
            "Please check the spelling or use a well-known city name."
        )

    top = results[0]
    return {
        "name":      top["name"],
        "latitude":  top["latitude"],
        "longitude": top["longitude"],
        "country":   top.get("country", ""),
        "timezone":  top.get("timezone", "auto"),
    }


# ---------------------------------------------------------------------------
# Forecast / current weather
# ---------------------------------------------------------------------------

def fetch_forecast(
    latitude: float,
    longitude: float,
    timezone: str = "auto",
) -> dict:
    """
    Fetch today's weather and the past 3 days from the Open-Meteo Forecast API.

    Returns a dict with:
        today   : dict  — today's weather variables
        past_3d : list  — list of dicts for the last 3 completed days
        raw     : dict  — full API response payload
    """
    params = {
        "latitude":     latitude,
        "longitude":    longitude,
        "daily":        ",".join(DAILY_VARIABLES),
        "forecast_days": 1,   # today (may be partial for current day)
        "past_days":    3,    # 3 completed days before today
        "timezone":     timezone,
    }
    logger.debug(
        "Fetching forecast for (%.4f, %.4f)", latitude, longitude
    )
    resp = requests.get(FORECAST_URL, params=params, timeout=_REQUEST_TIMEOUT)
    resp.raise_for_status()
    payload = resp.json()

    daily = payload["daily"]
    times = daily["time"]          # e.g. ["2026-08-09", ..., "2026-08-12"]

    records = []
    for i, t in enumerate(times):
        records.append({
            "date":                     t,
            "temperature_2m_max":       daily["temperature_2m_max"][i],
            "temperature_2m_min":       daily["temperature_2m_min"][i],
            "relative_humidity_2m_max": daily["relative_humidity_2m_max"][i],
            "wind_speed_10m_max":       daily["wind_speed_10m_max"][i],
        })

    # Last record is today (forecast_days=1); the rest are past completed days
    today_record   = records[-1]
    past_3_records = records[:-1]   # up to 3 prior days

    return {
        "today":   today_record,
        "past_3d": past_3_records,
        "raw":     payload,
    }


def compute_rolling_avg(today_max: float, past_records: list[dict]) -> float:
    """
    Compute a 3-day rolling average of max temperature.

    Uses up to 3 prior days. Falls back gracefully if fewer days are available.
    The current day's temperature is included to form the window.
    """
    max_temps = [r["temperature_2m_max"] for r in past_records if r["temperature_2m_max"] is not None]
    # Include today in the 3-day window
    window = (max_temps + [today_max])[-3:]
    return round(sum(window) / len(window), 2)


def get_current_weather_and_features(city: str) -> dict:
    """
    High-level helper: geocode a city, fetch forecast, compute rolling average.

    Returns a dict with keys:
        city_info          : dict  — name, latitude, longitude, country, timezone
        today              : dict  — today's weather variables
        rolling_max_temp_3d: float — 3-day rolling avg of max temp
    """
    city_info = geocode_city(city)
    forecast  = fetch_forecast(
        city_info["latitude"],
        city_info["longitude"],
        city_info["timezone"],
    )
    rolling = compute_rolling_avg(
        today_max    = forecast["today"]["temperature_2m_max"],
        past_records = forecast["past_3d"],
    )
    return {
        "city_info":           city_info,
        "today":               forecast["today"],
        "rolling_max_temp_3d": rolling,
    }
