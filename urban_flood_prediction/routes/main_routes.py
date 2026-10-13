from flask import Blueprint, render_template
from config import DISTRICTS

main_bp = Blueprint("main", __name__)

@main_bp.route("/")
def home():
    return render_template("index.html", districts=list(DISTRICTS.keys()))

@main_bp.route("/dashboard")
def dashboard():
    return render_template("dashboard.html", districts=list(DISTRICTS.keys()))

@main_bp.route("/weather")
def weather_page():
    return render_template("weather.html", districts=list(DISTRICTS.keys()))

@main_bp.route("/prediction")
def prediction_page():
    return render_template("prediction.html", districts=list(DISTRICTS.keys()))

@main_bp.route("/alerts")
def alerts_page():
    return render_template("alerts.html")

@main_bp.route("/map")
def map_page():
    return render_template("map.html", districts=list(DISTRICTS.keys()))

@main_bp.route("/history")
def history_page():
    return render_template("history.html")

@main_bp.route("/rainfall-analysis")
def rainfall_analysis_page():
    return render_template("rainfall_analysis.html")

@main_bp.route("/how-it-works")
def how_it_works():
    return render_template("how_it_works.html")
