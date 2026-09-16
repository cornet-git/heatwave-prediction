"""
api.py — FastAPI application for the Climate Intelligence Heatwave Prediction service.

Endpoints
---------
POST /predict
    Accepts raw features (max_temp, humidity, wind_speed) OR a city name.
    Returns predicted risk level and confidence score.

GET /current-weather/{city}
    Fetches live weather for the city via Open-Meteo Forecast API,
    then returns current conditions + heatwave risk prediction.

GET /health
    Simple liveness check — also pre-warms the model cache.
"""

from __future__ import annotations

import traceback
from typing import Optional

import requests
from fastapi import APIRouter, FastAPI, HTTPException, Path
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field, field_validator, model_validator

from src.predictor import predict
from src.utils import PROJECT_ROOT, get_logger
from src.weather import CityNotFoundError, get_current_weather_and_features

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# App initialisation
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Climate Intelligence — Heatwave Risk Prediction API",
    description=(
        "Predicts daily heatwave risk level (High / Moderate / Low) using a "
        "trained Random Forest classifier. Supports raw feature input or "
        "live weather lookup by city name."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS — allow all origins so any local frontend (or tools like Swagger) can call this
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------

class PredictRequest(BaseModel):
    """
    Accept either a city name OR raw weather features — not both, not neither.

    Examples
    --------
    City-based   : {"city": "Mumbai"}
    Feature-based: {"max_temp": 42.0, "humidity": 60.0, "wind_speed": 15.0}
    Full features: {"max_temp": 42.0, "humidity": 60.0, "wind_speed": 15.0,
                    "rolling_max_temp_3d": 39.5}
    """

    city: Optional[str] = Field(
        None,
        description="City name (geocoded via Open-Meteo). Provide this OR raw features.",
        examples=["Mumbai", "Pune", "Delhi"],
    )
    max_temp: Optional[float] = Field(
        None,
        description="Daily maximum air temperature (°C).",
        examples=[42.0],
    )
    humidity: Optional[float] = Field(
        None,
        ge=0,
        le=100,
        description="Daily maximum relative humidity (0–100 %).",
        examples=[65.0],
    )
    wind_speed: Optional[float] = Field(
        None,
        ge=0,
        description="Daily maximum wind speed (km/h).",
        examples=[18.5],
    )
    rolling_max_temp_3d: Optional[float] = Field(
        None,
        description=(
            "3-day rolling mean of max temperature (°C). "
            "Computed automatically when city is provided. "
            "Falls back to max_temp when omitted in feature mode."
        ),
        examples=[39.5],
    )

    @field_validator("city")
    @classmethod
    def city_must_be_non_empty(cls, v: Optional[str]) -> Optional[str]:
        if v is not None and not v.strip():
            raise ValueError("city must not be blank or whitespace.")
        return v.strip() if v else v

    @model_validator(mode="after")
    def check_mutual_exclusivity(self) -> "PredictRequest":
        has_city     = self.city is not None
        raw_fields   = [self.max_temp, self.humidity, self.wind_speed]
        has_features = all(f is not None for f in raw_fields)
        any_features = any(f is not None for f in raw_fields)

        if has_city and any_features:
            raise ValueError(
                "Provide either 'city' OR raw feature fields "
                "(max_temp, humidity, wind_speed) — not both."
            )
        if not has_city and not has_features:
            raise ValueError(
                "Provide either 'city' or all three feature fields: "
                "max_temp, humidity, wind_speed."
            )
        if not has_city and any_features and not has_features:
            missing = [
                name for name, val in [
                    ("max_temp", self.max_temp),
                    ("humidity", self.humidity),
                    ("wind_speed", self.wind_speed),
                ]
                if val is None
            ]
            raise ValueError(
                f"Missing required feature field(s): {', '.join(missing)}."
            )
        return self

    model_config = {
        "json_schema_extra": {
            "examples": [
                {"city": "Mumbai"},
                {"max_temp": 42.5, "humidity": 60.0, "wind_speed": 18.3},
                {
                    "max_temp": 42.5,
                    "humidity": 60.0,
                    "wind_speed": 18.3,
                    "rolling_max_temp_3d": 39.0,
                },
            ]
        }
    }


class PredictionResult(BaseModel):
    risk_level: str    = Field(..., description="Predicted risk: High | Moderate | Low")
    confidence: float  = Field(..., description="Probability of predicted class [0, 1]")
    probabilities: dict[str, float] = Field(..., description="Per-class probabilities")
    features_used: dict[str, float] = Field(..., description="Feature vector sent to model")


class PredictResponse(BaseModel):
    source: str = Field(..., description="'city' | 'raw_features'")
    city_info: Optional[dict] = Field(None, description="Resolved city details (city mode only)")
    prediction: PredictionResult


class WeatherData(BaseModel):
    date: str
    temperature_2m_max: Optional[float]
    temperature_2m_min: Optional[float]
    relative_humidity_2m_max: Optional[float]
    wind_speed_10m_max: Optional[float]
    rolling_max_temp_3d: float


class CurrentWeatherResponse(BaseModel):
    city: str
    country: str
    latitude: float
    longitude: float
    weather: WeatherData
    prediction: PredictionResult


# ---------------------------------------------------------------------------
# Exception handlers
# ---------------------------------------------------------------------------

@app.exception_handler(CityNotFoundError)
async def city_not_found_handler(request, exc: CityNotFoundError):
    return JSONResponse(
        status_code=404,
        content={"detail": str(exc), "error_type": "city_not_found"},
    )


@app.exception_handler(FileNotFoundError)
async def model_not_found_handler(request, exc: FileNotFoundError):
    return JSONResponse(
        status_code=503,
        content={
            "detail": str(exc),
            "error_type": "model_unavailable",
            "hint": "Run `python main.py` to train and save the model first.",
        },
    )


@app.exception_handler(requests.RequestException)
async def upstream_error_handler(request, exc: requests.RequestException):
    return JSONResponse(
        status_code=502,
        content={
            "detail": f"Upstream weather API error: {exc}",
            "error_type": "upstream_api_error",
        },
    )


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

router = APIRouter()


@router.get("/health", tags=["Utility"])
async def health():
    """
    Liveness check. Also pre-warms the model cache on first call.
    """
    try:
        from src.predictor import _load_model
        _load_model()
        model_ready = True
    except FileNotFoundError:
        model_ready = False

    return {
        "status":      "ok" if model_ready else "degraded",
        "model_ready": model_ready,
        "message": (
            "Model loaded and ready."
            if model_ready
            else "model.pkl not found — run `python main.py` first."
        ),
    }


@router.post(
    "/predict",
    response_model=PredictResponse,
    tags=["Prediction"],
    summary="Predict heatwave risk from raw features or a city name",
)
async def predict_risk(body: PredictRequest):
    """
    Predict the heatwave risk level for a given day.

    **Input options (mutually exclusive):**

    - **City mode** — provide `city`. Live weather is fetched automatically
      and the 3-day rolling average is computed from the last 3 days.
    - **Feature mode** — provide `max_temp`, `humidity`, `wind_speed`.
      Optionally provide `rolling_max_temp_3d`; if omitted it defaults to
      `max_temp` (conservative same-day assumption).

    **Response:**
    Returns `risk_level` (High / Moderate / Low), `confidence`, per-class
    `probabilities`, and the exact `features_used` for transparency.
    """
    try:
        # ---------------------------------------------------------------- city mode
        if body.city:
            logger.info("Predict request (city mode): city=%s", body.city)
            data = get_current_weather_and_features(body.city)

            today   = data["today"]
            rolling = data["rolling_max_temp_3d"]

            result = predict(
                max_temp            = today["temperature_2m_max"],
                humidity            = today["relative_humidity_2m_max"],
                wind_speed          = today["wind_speed_10m_max"],
                rolling_max_temp_3d = rolling,
            )
            ci = data["city_info"]
            return PredictResponse(
                source    = "city",
                city_info = {
                    "name":      ci["name"],
                    "country":   ci["country"],
                    "latitude":  ci["latitude"],
                    "longitude": ci["longitude"],
                },
                prediction = PredictionResult(**result),
            )

        # -------------------------------------------------------------- feature mode
        logger.info(
            "Predict request (feature mode): max_temp=%.1f, humidity=%.1f, wind=%.1f",
            body.max_temp, body.humidity, body.wind_speed,
        )
        result = predict(
            max_temp            = body.max_temp,
            humidity            = body.humidity,
            wind_speed          = body.wind_speed,
            rolling_max_temp_3d = body.rolling_max_temp_3d,
        )
        return PredictResponse(
            source     = "raw_features",
            city_info  = None,
            prediction = PredictionResult(**result),
        )

    except CityNotFoundError:
        raise   # handled by the registered exception handler
    except FileNotFoundError:
        raise   # handled by the registered exception handler
    except requests.RequestException:
        raise   # handled by the registered exception handler
    except Exception as exc:
        logger.error("Unexpected error in /predict: %s", traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Internal error: {exc}")


@router.get(
    "/current-weather/{city}",
    response_model=CurrentWeatherResponse,
    tags=["Weather"],
    summary="Get today's weather and predicted risk for a city",
)
async def current_weather(
    city: str = Path(
        ...,
        min_length=1,
        description="City name (e.g. Mumbai, Pune, Delhi, Chennai)",
        examples=["Mumbai"],
    ),
):
    """
    Fetch **today's live weather** for the given city (via Open-Meteo Forecast API)
    and return both current conditions and the predicted heatwave risk level.

    The 3-day rolling average of max temperature is computed from the last 3
    completed days, ensuring an accurate spike-detection signal.

    **Path parameter:**  
    `city` — e.g. `Mumbai`, `Pune`, `Delhi`. Must be a non-empty string.
    """
    city = city.strip()
    if not city:
        raise HTTPException(
            status_code=422,
            detail="City name must not be blank or whitespace.",
        )

    try:
        logger.info("Current-weather request: city=%s", city)
        data = get_current_weather_and_features(city)

        today   = data["today"]
        rolling = data["rolling_max_temp_3d"]
        ci      = data["city_info"]

        result = predict(
            max_temp            = today["temperature_2m_max"],
            humidity            = today["relative_humidity_2m_max"],
            wind_speed          = today["wind_speed_10m_max"],
            rolling_max_temp_3d = rolling,
        )

        return CurrentWeatherResponse(
            city      = ci["name"],
            country   = ci["country"],
            latitude  = ci["latitude"],
            longitude = ci["longitude"],
            weather   = WeatherData(
                date                      = today["date"],
                temperature_2m_max        = today["temperature_2m_max"],
                temperature_2m_min        = today["temperature_2m_min"],
                relative_humidity_2m_max  = today["relative_humidity_2m_max"],
                wind_speed_10m_max        = today["wind_speed_10m_max"],
                rolling_max_temp_3d       = rolling,
            ),
            prediction = PredictionResult(**result),
        )

    except CityNotFoundError:
        raise   # handled by the registered exception handler
    except FileNotFoundError:
        raise   # handled by the registered exception handler
    except requests.RequestException:
        raise   # handled by the registered exception handler
    except Exception as exc:
        logger.error(
            "Unexpected error in /current-weather/%s: %s", city, traceback.format_exc()
        )
        raise HTTPException(status_code=500, detail=f"Internal error: {exc}")


# ---------------------------------------------------------------------------
# Root and router mounting
# ---------------------------------------------------------------------------

@app.get("/", include_in_schema=False)
async def serve_frontend():
    """Serve the single-page application frontend."""
    index_file = PROJECT_ROOT / "index.html"
    if index_file.exists():
        return FileResponse(index_file)
    return JSONResponse({
        "status": "ok",
        "service": "Climate Intelligence — Heatwave Risk Prediction API",
        "docs": "/docs",
    })


# Include routes at root (e.g. /predict, /current-weather) and under /api
app.include_router(router)
app.include_router(router, prefix="/api")

