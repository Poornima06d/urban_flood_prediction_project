
import mysql.connector
from config import DB_CONFIG


def get_connection():
    return mysql.connector.connect(**DB_CONFIG)


def save_prediction(
    district,
    probability,
    risk_level,
    rainfall=None,
    temperature=None,
    humidity=None,
    latitude=None,
    longitude=None,
    wind_speed=None,
    pressure=None,
):
    conn = get_connection()
    cur = conn.cursor()

    # Save prediction using the existing database column names
    cur.execute(
        """
        INSERT INTO predictions
        (
            location,
            latitude,
            longitude,
            temperature,
            humidity,
            rainfall,
            wind_speed,
            pressure,
            probability,
            risk_level
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """,
        (
            district,
            latitude,
            longitude,
            temperature,
            humidity,
            rainfall,
            wind_speed,
            pressure,
            probability,
            risk_level,
        ),
    )

    prediction_id = cur.lastrowid

    # Save alert using the existing alerts table
    cur.execute(
        """
        INSERT INTO alerts
        (district, risk_level, message)
        VALUES (%s, %s, %s)
        """,
        (
            district,
            risk_level,
            build_alert_message(district, risk_level, probability),
        ),
    )

    conn.commit()

    cur.close()
    conn.close()

    return prediction_id


def build_alert_message(district, risk_level, probability):
    if risk_level == "High":
        return (
            f"High flood risk expected in {district} "
            f"within the next 24 hours."
        )

    if risk_level == "Medium":
        return (
            f"Moderate flood risk expected in {district} "
            f"within the next 24 hours. Monitor conditions."
        )

    return (
        f"Low flood risk expected in {district} "
        f"within the next 24 hours."
    )


def fetch_alerts():
    conn = get_connection()
    cur = conn.cursor(dictionary=True)

    cur.execute(
        """
        SELECT id, district, risk_level, message, created_at
        FROM alerts
        ORDER BY created_at DESC
        LIMIT 100
        """
    )

    rows = cur.fetchall()

    cur.close()
    conn.close()

    for r in rows:
        if r["created_at"]:
            r["created_at"] = r["created_at"].isoformat()

    return rows


def fetch_latest_predictions():
    conn = get_connection()
    cur = conn.cursor(dictionary=True)

    cur.execute(
        """
        SELECT
            p.id,
            p.location,
            p.latitude,
            p.longitude,
            p.temperature,
            p.humidity,
            p.rainfall,
            p.wind_speed,
            p.pressure,
            p.probability,
            p.risk_level,
            p.created_at
        FROM predictions p
        INNER JOIN (
            SELECT location, MAX(id) AS max_id
            FROM predictions
            GROUP BY location
        ) x
        ON p.id = x.max_id
        ORDER BY p.created_at DESC
        """
    )

    rows = cur.fetchall()

    cur.close()
    conn.close()

    for r in rows:
        if r["created_at"]:
            r["created_at"] = r["created_at"].isoformat()

    return rows