from flask import Blueprint, jsonify
from database.db import fetch_alerts

alert_bp = Blueprint("alerts", __name__)

@alert_bp.route("/alerts", methods=["GET"])
def alerts():
    try:
        return jsonify(fetch_alerts())
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500
