
import os
import joblib
import pandas as pd

from config import MODEL_PATH
from database.db import save_prediction
from services.weather_api import get_weather
from services.feature_engineering import create_features, FEATURES


MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "model",
    "xgboost_flood_24h_model_v5.pkl",
)

BEST_THRESHOLD = 0.50
HIGH_THRESHOLD = 0.70


def load_model():
    if not os.path.exists(MODEL_PATH):
        raise RuntimeError(
            f"Trained model not found: {MODEL_PATH}"
        )

    return joblib.load(MODEL_PATH)


def classify_risk(probability):
    if probability < BEST_THRESHOLD:
        return "Low"
    elif probability < HIGH_THRESHOLD:
        return "Medium"
    else:
        return "High"


def predict_risk(district):
    # Get current weather from OpenWeather
    weather = get_weather(district)

    # Use rainfall from current weather
    rainfall = weather.get("rainfall_mm", 0.0)

    # Create the 29 features expected by V5 model
    features = create_features(
        district=district,
        rainfall_mm=rainfall,
        prediction_date=pd.Timestamp.now(),
    )

    # Maintain exact feature order used during training
    X = pd.DataFrame(
        [[features[feature] for feature in FEATURES]],
        columns=FEATURES,
    )

    # Load trained XGBoost model
    model = load_model()

    # Flood probability
    probability = float(model.predict_proba(X)[0][1])

    # Risk classification
    risk_level = classify_risk(probability)

    # Save prediction and alert to MySQL
    prediction_id = save_prediction(
        district=district,
        probability=probability,
        risk_level=risk_level,
        rainfall=rainfall,
        temperature=weather.get("temperature_c"),
	humidity=weather.get("humidity_pct"),
        wind_speed=weather.get("wind_speed_mps"),
	pressure=weather.get("pressure_hpa"),
    )

    warning = build_warning(district, risk_level, probability)

    return {
        "prediction_id": prediction_id,
        "district": district,
        "probability": round(probability, 4),
        "risk_level": risk_level,
        "threshold": BEST_THRESHOLD,
        "warning": warning,
        "weather": weather,
        "features": features,
        "model_features": FEATURES,
        "feature_count": len(FEATURES),
        "model": "XGBoost V5",
    }


def build_warning(district, risk_level, probability):
    probability_percent = round(probability * 100, 2)

    if risk_level == "High":
        return (
            f"High flood risk indicated for {district}. "
            f"Predicted probability: {probability_percent}%. "
            f"Monitor conditions and follow official warnings."
        )

    if risk_level == "Medium":
        return (
            f"Moderate flood risk indicated for {district}. "
            f"Predicted probability: {probability_percent}%. "
            f"Continue monitoring weather conditions."
        )

    return (
        f"Low flood risk indicated for {district}. "
        f"Predicted probability: {probability_percent}%."
    )

