from flask import Blueprint, jsonify
from database.db import fetch_latest_predictions

map_bp = Blueprint("map", __name__)

@map_bp.route("/map-data", methods=["GET"])
def map_data():
    try:
        return jsonify(fetch_latest_predictions())
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500
