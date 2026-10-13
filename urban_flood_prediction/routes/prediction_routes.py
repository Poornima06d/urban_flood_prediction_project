from flask import Blueprint, request, jsonify

from services.prediction_service import predict_risk
from services.weather_api import (
    get_weather,
    get_weather_forecast,
)


prediction_bp = Blueprint(
    "prediction",
    __name__
)


@prediction_bp.route(
    "/weather",
    methods=["GET"]
)
def weather():

    district = request.args.get(
        "district",
        ""
    )

    try:
        return jsonify(
            get_weather(district)
        )

    except Exception as exc:

        return jsonify({
            "error": str(exc)
        }), 400


@prediction_bp.route(
    "/forecast",
    methods=["GET"]
)
def forecast():

    district = request.args.get(
        "district",
        ""
    )

    try:
        return jsonify(
            get_weather_forecast(
                district
            )
        )

    except Exception as exc:

        return jsonify({
            "error": str(exc)
        }), 400


@prediction_bp.route(
    "/predict",
    methods=["POST"]
)
def predict():

    payload = request.get_json(
        silent=True
    ) or {}

    district = payload.get(
        "district",
        ""
    )

    try:

        result = predict_risk(
            district
        )

        return jsonify(result)

    except Exception as exc:

        return jsonify({
            "error": str(exc)
        }), 400
