import os
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "xgboost_flood_24h_model_v5.pkl")
MODEL_META_PATH = os.path.join(BASE_DIR, "model", "model_metadata.joblib")

OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY", "")

DB_CONFIG = {
    "host": os.getenv("DB_HOST", "localhost"),
    "port": int(os.getenv("DB_PORT", "3306")),
    "database": os.getenv("DB_NAME", "urban_flood"),
    "user": os.getenv("DB_USER", "root"),
    "password": os.getenv("DB_PASSWORD", ""),
}

DISTRICTS = {
    "Belagavi": {"lat": 15.8497, "lon": 74.4977},
    "Bagalkot": {"lat": 16.1691, "lon": 75.6615},
    "Shivamogga": {"lat": 13.9299, "lon": 75.5681},
    "Chikkamagaluru": {"lat": 13.3153, "lon": 75.7754},
    "Uttara Kannada": {"lat": 14.7952, "lon": 74.6869},
}
