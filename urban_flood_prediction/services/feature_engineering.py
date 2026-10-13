import os
import numpy as np
import pandas as pd


DATA_PATH = "data/processed/flood_training_24h.csv"

DISTRICT_CODES = {
    "Belagavi": 1,
    "Bagalkot": 2,
    "Shivamogga": 3,
    "Chikkamagaluru": 4,
    "Uttara Kannada": 5,
}

BASE_FEATURES = [
    "Rainfall_mm",
    "Rain_lag1",
    "Rain_2d_prev",
    "Rain_3d_prev",
    "Rain_5d_prev",
    "Rain_7d_prev",
    "Rain_10d_prev",
    "Rain_15d_prev",
    "Rain_30d_prev",
    "Rain_max_3d_prev",
    "Rain_max_7d_prev",
    "Rain_max_15d_prev",
    "Rain_max_30d_prev",
    "Rain_nonzero_7d",
    "Rain_nonzero_15d",
    "Rain_nonzero_30d",
    "Rain_7d_avg_prev",
    "Rain_30d_avg_prev",
    "Rain_trend",
    "Month",
    "DayOfYear",
    "District_Code",
]

NORMALIZED_FEATURES = [
    "Rainfall_ratio_district",
    "Rain_lag1_ratio_district",
    "Rain_7d_ratio_district",
    "Rain_30d_ratio_district",
    "Rainfall_zscore_district",
    "Rain_7d_accum_ratio_district",
    "Rain_30d_accum_ratio_district",
]

FEATURES = BASE_FEATURES + NORMALIZED_FEATURES


def load_training_data():
    if not os.path.exists(DATA_PATH):
        raise RuntimeError(
            f"Training dataset not found: {DATA_PATH}"
        )

    df = pd.read_csv(DATA_PATH)
    df["Date"] = pd.to_datetime(df["Date"])

    return df


def calculate_district_baselines(df):
    """
    Reproduce the exact V5 baseline calculation.

    IMPORTANT:
    Only data from the training period (<= 2014)
    is used, exactly as in train_model.py.
    """

    train_df = df[
        df["Date"].dt.year <= 2014
    ].copy()

    baseline = (
        train_df
        .groupby("District")["Rainfall_mm"]
        .agg(
            Rain_mean="mean",
            Rain_std="std"
        )
        .reset_index()
    )

    baseline["Rain_mean"] = baseline["Rain_mean"].replace(
        0, np.nan
    )

    baseline["Rain_std"] = baseline["Rain_std"].replace(
        0, np.nan
    )

    return baseline


def get_latest_historical_features(district):
    """
    Gets the latest available row for the selected district.

    The processed dataset currently ends in 2023.
    Therefore this is historical baseline information,
    not a claim that it represents today's rainfall.
    """

    df = load_training_data()

    district_df = df[
        df["District"] == district
    ].sort_values("Date")

    if district_df.empty:
        raise ValueError(
            f"No historical data found for {district}."
        )

    return district_df.iloc[-1].copy()


def create_features(
    district,
    rainfall_mm=None,
    historical_values=None,
    prediction_date=None
):
    """
    Create exactly the 29 features expected by V5.

    historical_values can optionally contain recent rainfall
    feature values supplied by the dashboard/API.

    If values are not supplied, the latest available processed
    historical row is used.
    """

    if district not in DISTRICT_CODES:
        raise ValueError("Invalid district.")

    df = load_training_data()

    latest = get_latest_historical_features(district)

    baseline = calculate_district_baselines(df)

    district_baseline = baseline[
        baseline["District"] == district
    ]

    if district_baseline.empty:
        raise ValueError(
            f"No district baseline found for {district}."
        )

    rain_mean = float(
        district_baseline.iloc[0]["Rain_mean"]
    )

    rain_std = float(
        district_baseline.iloc[0]["Rain_std"]
    )

    # Start with historical values
    values = {}

    for feature in BASE_FEATURES:
        if feature in latest.index:
            values[feature] = latest[feature]

    # Current rainfall can override the historical Rainfall_mm.
    if rainfall_mm is not None:
        values["Rainfall_mm"] = float(rainfall_mm)

    # Explicit API/dashboard values override historical values.
    if historical_values:
        for feature in BASE_FEATURES:
            if feature in historical_values:
                values[feature] = historical_values[feature]

    # Date information
    if prediction_date is None:
        prediction_date = pd.Timestamp.now()

    prediction_date = pd.Timestamp(prediction_date)

    values["Month"] = prediction_date.month
    values["DayOfYear"] = prediction_date.dayofyear
    values["District_Code"] = DISTRICT_CODES[district]

    # Make sure every base feature exists.
    for feature in BASE_FEATURES:
        if feature not in values or pd.isna(values[feature]):
            values[feature] = 0.0

    # Convert numerical fields.
    for feature in BASE_FEATURES:
        if feature not in ["Month", "DayOfYear", "District_Code"]:
            values[feature] = float(values[feature])

    values["Month"] = int(values["Month"])
    values["DayOfYear"] = int(values["DayOfYear"])
    values["District_Code"] = int(values["District_Code"])

    # --------------------------------------------------------
    # EXACT V5 NORMALIZED FEATURES
    # --------------------------------------------------------

    values["Rainfall_ratio_district"] = (
        values["Rainfall_mm"] / rain_mean
    )

    values["Rain_lag1_ratio_district"] = (
        values["Rain_lag1"] / rain_mean
    )

    values["Rain_7d_ratio_district"] = (
        values["Rain_7d_avg_prev"] / rain_mean
    )

    values["Rain_30d_ratio_district"] = (
        values["Rain_30d_avg_prev"] / rain_mean
    )

    values["Rainfall_zscore_district"] = (
        (values["Rainfall_mm"] - rain_mean) /
        rain_std
    )

    values["Rain_7d_accum_ratio_district"] = (
        values["Rain_7d_prev"] /
        (rain_mean * 7)
    )

    values["Rain_30d_accum_ratio_district"] = (
        values["Rain_30d_prev"] /
        (rain_mean * 30)
    )

    # Same protection used by V5 training.
    for feature in NORMALIZED_FEATURES:
        if not np.isfinite(values[feature]):
            values[feature] = 0.0

    # Return exactly the model's 29 features.
    return {
        feature: values[feature]
        for feature in FEATURES
    }