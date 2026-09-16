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
_HEADERS = {
    "User-Agent": "HeatWatch-HeatwaveMonitor/1.0 (ClimateIntelligence; https://github.com)"
}

# Fast-path & offline/fallback coordinates for common cities
_KNOWN_CITIES: dict[str, dict] = {
    "mumbai": {"name": "Mumbai", "latitude": 19.0760, "longitude": 72.8777, "country": "India", "timezone": "Asia/Kolkata"},
    "pune": {"name": "Pune", "latitude": 18.5204, "longitude": 73.8567, "country": "India", "timezone": "Asia/Kolkata"},
    "delhi": {"name": "Delhi", "latitude": 28.6139, "longitude": 77.2090, "country": "India", "timezone": "Asia/Kolkata"},
    "new delhi": {"name": "New Delhi", "latitude": 28.6139, "longitude": 77.2090, "country": "India", "timezone": "Asia/Kolkata"},
    "bengaluru": {"name": "Bengaluru", "latitude": 12.9716, "longitude": 77.5946, "country": "India", "timezone": "Asia/Kolkata"},
    "bangalore": {"name": "Bengaluru", "latitude": 12.9716, "longitude": 77.5946, "country": "India", "timezone": "Asia/Kolkata"},
    "hyderabad": {"name": "Hyderabad", "latitude": 17.3850, "longitude": 78.4867, "country": "India", "timezone": "Asia/Kolkata"},
    "ahmedabad": {"name": "Ahmedabad", "latitude": 23.0225, "longitude": 72.5714, "country": "India", "timezone": "Asia/Kolkata"},
    "chennai": {"name": "Chennai", "latitude": 13.0827, "longitude": 80.2707, "country": "India", "timezone": "Asia/Kolkata"},
    "kolkata": {"name": "Kolkata", "latitude": 22.5726, "longitude": 88.3639, "country": "India", "timezone": "Asia/Kolkata"},
    "surat": {"name": "Surat", "latitude": 21.1702, "longitude": 72.8311, "country": "India", "timezone": "Asia/Kolkata"},
    "jaipur": {"name": "Jaipur", "latitude": 26.9124, "longitude": 75.7873, "country": "India", "timezone": "Asia/Kolkata"},
    "lucknow": {"name": "Lucknow", "latitude": 26.8467, "longitude": 80.9462, "country": "India", "timezone": "Asia/Kolkata"},
    "nagpur": {"name": "Nagpur", "latitude": 21.1458, "longitude": 79.0882, "country": "India", "timezone": "Asia/Kolkata"},
    "indore": {"name": "Indore", "latitude": 22.7196, "longitude": 75.8577, "country": "India", "timezone": "Asia/Kolkata"},
    "bhopal": {"name": "Bhopal", "latitude": 23.2599, "longitude": 77.4126, "country": "India", "timezone": "Asia/Kolkata"},
    "patna": {"name": "Patna", "latitude": 25.5941, "longitude": 85.1376, "country": "India", "timezone": "Asia/Kolkata"},
    "vadodara": {"name": "Vadodara", "latitude": 22.3072, "longitude": 73.1812, "country": "India", "timezone": "Asia/Kolkata"},
    "nashik": {"name": "Nashik", "latitude": 19.9975, "longitude": 73.7898, "country": "India", "timezone": "Asia/Kolkata"},
}


# ---------------------------------------------------------------------------
# Geocoding
# ---------------------------------------------------------------------------

class CityNotFoundError(ValueError):
    """Raised when the geocoding API returns no results for a city name."""


def geocode_city(city: str) -> dict:
    """
    Resolve a city name to coordinates via fast cache or Open-Meteo Geocoding API.
    """
    normalized = city.strip().lower()

    # Fast-path check for well-known cities
    if normalized in _KNOWN_CITIES:
        logger.debug("Resolved '%s' via known cities cache", city)
        return dict(_KNOWN_CITIES[normalized])

    params = {
        "name":     city.strip(),
        "count":    1,
        "language": "en",
        "format":   "json",
    }
    logger.debug("Geocoding city via Open-Meteo: %s", city)
    try:
        resp = requests.get(
            GEOCODING_URL,
            params=params,
            headers=_HEADERS,
            timeout=_REQUEST_TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()
    except requests.RequestException as exc:
        # Fallback check before raising
        if normalized in _KNOWN_CITIES:
            return dict(_KNOWN_CITIES[normalized])
        logger.warning("Geocoding API network error: %s", exc)
        raise

    if isinstance(data, dict) and data.get("error"):
        reason = data.get("reason", "Unknown API error")
        raise requests.RequestException(f"Geocoding service error: {reason}")

    results = data.get("results") if isinstance(data, dict) else None
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
    resp = requests.get(
        FORECAST_URL,
        params=params,
        headers=_HEADERS,
        timeout=_REQUEST_TIMEOUT,
    )
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
