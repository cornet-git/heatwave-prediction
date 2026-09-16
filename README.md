# Heatwave Risk Prediction Model

> **Part of: Climate Intelligence for Heatwave Monitoring, Prediction, and Early Warning**

A Python machine-learning project that fetches five years of real historical daily weather data for **Mumbai** and **Pune**, engineers heatwave-relevant features, labels each day with a risk tier, and trains a **Random Forest classifier** to predict heatwave risk (`High / Moderate / Low`).

---

## Project Structure

```
heatwave_prediction/
├── data/                        # Auto-created on first run
│   ├── weather_raw.csv          # Raw API data
│   └── weather_processed.csv   # Engineered features + labels
├── models/                      # Auto-created on first run
│   ├── model.pkl                # Serialised RandomForestClassifier
│   └── features.json            # Input schema for inference
├── src/
│   ├── __init__.py
│   ├── utils.py                 # Logging + path constants
│   ├── data_fetch.py            # Open-Meteo API client
│   ├── feature_engineering.py  # Feature engineering + labelling
│   └── train.py                 # Training, evaluation, export
├── main.py                      # Pipeline orchestrator
├── requirements.txt
└── README.md
```

---

## Data Source

[Open-Meteo Historical Weather API](https://open-meteo.com/en/docs/historical-weather-api) — **free, no API key required**.

| Variable | Description | Unit |
|---|---|---|
| `temperature_2m_max` | Daily maximum air temperature at 2 m | °C |
| `temperature_2m_min` | Daily minimum air temperature at 2 m | °C |
| `relative_humidity_2m_max` | Daily maximum relative humidity at 2 m | % |
| `wind_speed_10m_max` | Daily maximum wind speed at 10 m | km/h |

Cities covered: **Mumbai** (19.076°N, 72.877°E) and **Pune** (18.5204°N, 73.8567°E).

---

## Feature Engineering

| Feature | Description |
|---|---|
| `temperature_2m_max` | Raw daily max temperature |
| `relative_humidity_2m_max` | Raw daily max humidity |
| `wind_speed_10m_max` | Raw daily max wind speed |
| `rolling_max_temp_3d` | 3-day rolling mean of max temperature (per city) |

---

## Labelling Rules

Labels are assigned in **priority order**:

| Priority | Condition | Label |
|---|---|---|
| 1 | `max_temp ≥ 40°C` **OR** `(max_temp − rolling_avg) ≥ 4.5°C` | **High** |
| 2 | `35°C ≤ max_temp < 40°C` | **Moderate** |
| 3 | `max_temp < 35°C` | **Low** |

---

## Model

- **Algorithm**: `RandomForestClassifier` (scikit-learn)
- **Estimators**: 200 trees
- **Class weighting**: `balanced` (handles imbalanced classes)
- **Split**: 80 % train / 20 % test (stratified)
- **Evaluation**: Accuracy + per-class Precision, Recall, F1-score

---

## Setup & Usage

### 1. Create a virtual environment (recommended)

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the full pipeline

```bash
python main.py
```

This will:
1. Fetch ~3 650 records from the Open-Meteo API (~30 s)
2. Engineer features and label each day
3. Train the classifier and print the evaluation report
4. Save `models/model.pkl` and `models/features.json`

---

## Inference Example

```python
import joblib, json, numpy as np

clf = joblib.load("models/model.pkl")

# Feature order (from features.json):
# [temperature_2m_max, relative_humidity_2m_max, wind_speed_10m_max, rolling_max_temp_3d]
sample = np.array([[42.1, 55.0, 18.3, 38.5]])   # a hot Mumbai day
print(clf.predict(sample))          # e.g. ['High']
print(clf.predict_proba(sample))    # probability per class
```

---

## Output Files

| File | Description |
|---|---|
| `data/weather_raw.csv` | Raw daily weather from API |
| `data/weather_processed.csv` | Engineered features + risk labels |
| `models/model.pkl` | Serialised trained classifier |
| `models/features.json` | JSON schema describing expected model inputs |

---

## Running the Web Application Locally

1. Start the server:
   ```bash
   python run_server.py
   # Or on Windows, double-click start_server.bat
   ```
2. Open your browser and navigate to:
   - **Web App**: [http://localhost:8000](http://localhost:8000)
   - **Interactive API Docs (Swagger)**: [http://localhost:8000/docs](http://localhost:8000/docs)

---

## Deploying to Vercel

This repository is pre-configured with `vercel.json` and `api/index.py` for direct deployment:
1. Push your code to GitHub (see deployment guide).
2. Go to [vercel.com](https://vercel.com) and click **"Add New Project"**.
3. Import your GitHub repository.
4. Keep the default settings and click **Deploy**.

