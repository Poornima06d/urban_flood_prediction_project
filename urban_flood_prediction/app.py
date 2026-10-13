from flask import Flask
from routes.main_routes import main_bp
from routes.prediction_routes import prediction_bp
from routes.alert_routes import alert_bp
from routes.map_routes import map_bp

app = Flask(__name__)
app.config["JSON_SORT_KEYS"] = False

app.register_blueprint(main_bp)
app.register_blueprint(prediction_bp, url_prefix="/api")
app.register_blueprint(alert_bp, url_prefix="/api")
app.register_blueprint(map_bp, url_prefix="/api")

if __name__ == "__main__":
    app.run(debug=True)
