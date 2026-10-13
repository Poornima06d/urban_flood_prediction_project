import requests

from config import OPENWEATHER_API_KEY, DISTRICTS


CURRENT_WEATHER_URL = (
    "https://api.openweathermap.org/data/2.5/weather"
)

FORECAST_URL = (
    "https://api.openweathermap.org/data/2.5/forecast"
)


def _validate_district(district):

    if district not in DISTRICTS:
        raise ValueError("Invalid district.")

    if not OPENWEATHER_API_KEY:
        raise ValueError(
            "OPENWEATHER_API_KEY is not configured."
        )


def get_weather(district):

    _validate_district(district)

    lat = DISTRICTS[district]["lat"]
    lon = DISTRICTS[district]["lon"]

    params = {
        "lat": lat,
        "lon": lon,
        "appid": OPENWEATHER_API_KEY,
        "units": "metric",
    }

    response = requests.get(
        CURRENT_WEATHER_URL,
        params=params,
        timeout=15,
    )

    response.raise_for_status()

    data = response.json()

    return {
        "district": district,
        "temperature_c": data["main"]["temp"],
        "humidity_pct": data["main"]["humidity"],
        "pressure_hpa": data["main"]["pressure"],
        "wind_speed_mps": data["wind"]["speed"],
        "weather": data["weather"][0]["description"],
        "rainfall_mm": data.get(
            "rain", {}
        ).get("1h", 0.0),
    }


def get_weather_forecast(district):

    _validate_district(district)

    lat = DISTRICTS[district]["lat"]
    lon = DISTRICTS[district]["lon"]

    params = {
        "lat": lat,
        "lon": lon,
        "appid": OPENWEATHER_API_KEY,
        "units": "metric",
    }

    response = requests.get(
        FORECAST_URL,
        params=params,
        timeout=15,
    )

    response.raise_for_status()

    data = response.json()

    forecast = []

    for item in data.get("list", []):

        rain = item.get(
            "rain",
            {}
        ).get(
            "3h",
            0.0
        )

        forecast.append({
            "datetime": item.get("dt_txt"),
            "temperature_c": item["main"]["temp"],
            "humidity_pct": item["main"]["humidity"],
            "pressure_hpa": item["main"]["pressure"],
            "wind_speed_mps": item["wind"]["speed"],
            "weather": item["weather"][0]["description"],
            "rainfall_3h_mm": rain,
        })

    return {
        "district": district,
        "forecast": forecast,
    }