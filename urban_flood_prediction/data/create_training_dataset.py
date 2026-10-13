import os
import pandas as pd
import numpy as np


# ============================================================
# CONFIGURATION
# ============================================================

RAINFALL_FILE = "data/raw/imd_training_dataset_prepared.csv"
FLOOD_FILE = "data/raw/India_Flood_Inventory_Karnataka_5_Districts_Clean.csv"
OUTPUT_FILE = "data/processed/flood_training_24h.csv"

# Five project districts
DISTRICT_MAP = {
    "Bagalkote": "Bagalkot",
    "Bagalkot": "Bagalkot",
    "Belgaum": "Belagavi",
    "Belagavi": "Belagavi",
    "Chikkamagaluru": "Chikkamagaluru",
    "Chikmagalur": "Chikkamagaluru",
    "Shivamogga": "Shivamogga",
    "Shimoga": "Shivamogga",
    "Uttara Kannada": "Uttara Kannada",
    "UttaraKannada": "Uttara Kannada",
}

DISTRICT_CODE = {
    "Bagalkot": 1,
    "Belagavi": 2,
    "Chikkamagaluru": 3,
    "Shivamogga": 4,
    "Uttara Kannada": 5,
}


# ============================================================
# 1. LOAD RAINFALL DATA
# ============================================================

print("=" * 70)
print("LOADING RAINFALL DATA")
print("=" * 70)

rain = pd.read_csv(RAINFALL_FILE)

print("Rainfall rows:", len(rain))
print("Rainfall columns:", list(rain.columns))


# ------------------------------------------------------------
# Handle date column
# ------------------------------------------------------------

if "Date" not in rain.columns:
    raise ValueError(
        "Expected 'Date' column in rainfall dataset."
    )

if "District" not in rain.columns:
    raise ValueError(
        "Expected 'District' column in rainfall dataset."
    )

if "Rainfall_mm" not in rain.columns:
    raise ValueError(
        "Expected 'Rainfall_mm' column in rainfall dataset."
    )


rain["Date"] = pd.to_datetime(
    rain["Date"],
    errors="coerce"
)

rain["District"] = (
    rain["District"]
    .astype(str)
    .str.strip()
    .replace(DISTRICT_MAP)
)

rain["Rainfall_mm"] = pd.to_numeric(
    rain["Rainfall_mm"],
    errors="coerce"
).fillna(0)


# Keep only project districts
rain = rain[
    rain["District"].isin(DISTRICT_CODE.keys())
].copy()


# ============================================================
# 2. CREATE DAILY RAINFALL DATA
# ============================================================

daily = (
    rain
    .groupby(["District", "Date"], as_index=False)
    ["Rainfall_mm"]
    .max()
)

daily = daily.sort_values(
    ["District", "Date"]
).reset_index(drop=True)


print("Daily rainfall rows:", len(daily))


# ============================================================
# 3. CALENDAR FEATURES
# ============================================================

daily["Year"] = daily["Date"].dt.year
daily["Month"] = daily["Date"].dt.month
daily["DayOfYear"] = daily["Date"].dt.dayofyear


# ============================================================
# 4. LEAKAGE-SAFE RAINFALL FEATURES
# ============================================================

g = daily.groupby("District")["Rainfall_mm"]

# Previous-day rainfall
daily["Rain_lag1"] = g.shift(1)

# Previous N-day accumulated rainfall
daily["Rain_2d_prev"] = (
    g.shift(1).rolling(2).sum()
    .reset_index(level=0, drop=True)
)

daily["Rain_3d_prev"] = (
    g.shift(1).rolling(3).sum()
    .reset_index(level=0, drop=True)
)

daily["Rain_5d_prev"] = (
    g.shift(1).rolling(5).sum()
    .reset_index(level=0, drop=True)
)

daily["Rain_7d_prev"] = (
    g.shift(1).rolling(7).sum()
    .reset_index(level=0, drop=True)
)

daily["Rain_10d_prev"] = (
    g.shift(1).rolling(10).sum()
    .reset_index(level=0, drop=True)
)

daily["Rain_15d_prev"] = (
    g.shift(1).rolling(15).sum()
    .reset_index(level=0, drop=True)
)

daily["Rain_30d_prev"] = (
    g.shift(1).rolling(30).sum()
    .reset_index(level=0, drop=True)
)


# ------------------------------------------------------------
# Previous rolling maximum rainfall
# ------------------------------------------------------------

daily["Rain_max_3d_prev"] = (
    g.shift(1).rolling(3).max()
    .reset_index(level=0, drop=True)
)

daily["Rain_max_7d_prev"] = (
    g.shift(1).rolling(7).max()
    .reset_index(level=0, drop=True)
)

daily["Rain_max_15d_prev"] = (
    g.shift(1).rolling(15).max()
    .reset_index(level=0, drop=True)
)

daily["Rain_max_30d_prev"] = (
    g.shift(1).rolling(30).max()
    .reset_index(level=0, drop=True)
)


# ------------------------------------------------------------
# Number of rainy days in previous periods
# ------------------------------------------------------------

rain_nonzero = (daily["Rainfall_mm"] > 0).astype(int)

daily["_Rain_nonzero"] = rain_nonzero

nz = daily.groupby("District")["_Rain_nonzero"]

daily["Rain_nonzero_7d"] = (
    nz.shift(1).rolling(7).sum()
    .reset_index(level=0, drop=True)
)

daily["Rain_nonzero_15d"] = (
    nz.shift(1).rolling(15).sum()
    .reset_index(level=0, drop=True)
)

daily["Rain_nonzero_30d"] = (
    nz.shift(1).rolling(30).sum()
    .reset_index(level=0, drop=True)
)


# ------------------------------------------------------------
# Previous average rainfall
# ------------------------------------------------------------

daily["Rain_7d_avg_prev"] = (
    g.shift(1).rolling(7).mean()
    .reset_index(level=0, drop=True)
)

daily["Rain_30d_avg_prev"] = (
    g.shift(1).rolling(30).mean()
    .reset_index(level=0, drop=True)
)


# ------------------------------------------------------------
# Rainfall trend
# ------------------------------------------------------------

daily["Rain_trend"] = (
    daily["Rain_7d_avg_prev"]
    - daily["Rain_30d_avg_prev"]
)


# Remove temporary column
daily.drop(
    columns=["_Rain_nonzero"],
    inplace=True
)


# Fill initial rolling values
feature_columns = [
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
]

daily[feature_columns] = (
    daily[feature_columns]
    .fillna(0)
)


# ============================================================
# 5. LOAD FLOOD INVENTORY
# ============================================================

print()
print("=" * 70)
print("LOADING FLOOD INVENTORY")
print("=" * 70)

flood = pd.read_csv(FLOOD_FILE)

print("Original cleaned flood records:", len(flood))


# ============================================================
# 6. NORMALIZE FLOOD DISTRICT NAMES
# ============================================================

if "District_Clean" not in flood.columns:
    raise ValueError(
        "Expected 'District_Clean' column in flood inventory."
    )

flood["District_Clean"] = (
    flood["District_Clean"]
    .astype(str)
    .str.strip()
    .replace(DISTRICT_MAP)
)


flood = flood[
    flood["District_Clean"].isin(
        DISTRICT_CODE.keys()
    )
].copy()


# ============================================================
# 7. PARSE FLOOD DATES
# ============================================================

flood["Start_Date"] = pd.to_datetime(
    flood["Start Date"],
    errors="coerce",
    dayfirst=True
)

flood["End_Date"] = pd.to_datetime(
    flood["End Date"],
    errors="coerce",
    dayfirst=True
)

flood["Duration_Source"] = pd.to_numeric(
    flood["Duration(Days)"],
    errors="coerce"
)


# ============================================================
# 8. VALIDATE FLOOD EVENTS
# ============================================================

flood["Calculated_Duration"] = (
    flood["End_Date"]
    - flood["Start_Date"]
).dt.days + 1


# A flood event is accepted only when:
#
# 1. Start and End dates are valid
# 2. End date is not before Start date
# 3. Calculated duration is reasonably close
#    to the source duration
#
# We allow a tolerance of 7 days.

valid_event = (
    flood["Start_Date"].notna()
    & flood["End_Date"].notna()
    & flood["Duration_Source"].notna()
    & (flood["Calculated_Duration"] > 0)
    & (
        (
            flood["Calculated_Duration"]
            - flood["Duration_Source"]
        ).abs()
        <= 7
    )
)

valid_flood = flood[valid_event].copy()


print("Valid flood events:", len(valid_flood))
print(
    "Rejected suspicious events:",
    len(flood) - len(valid_flood)
)


# ============================================================
# 9. SHOW REJECTED EVENTS
# ============================================================

rejected = flood[~valid_event].copy()

if len(rejected) > 0:

    print()
    print("Rejected suspicious flood records:")

    print(
        rejected[
            [
                "UEI",
                "Start_Date",
                "End_Date",
                "Duration_Source",
                "Calculated_Duration",
                "District_Clean",
            ]
        ]
        .sort_values("Start_Date")
        .to_string(index=False)
    )


# ============================================================
# 10. CREATE FLOOD-DAY LABELS
# ============================================================

print()
print("=" * 70)
print("CREATING FLOOD LABELS")
print("=" * 70)

flood_days = []

for _, row in valid_flood.iterrows():

    start = row["Start_Date"]
    end = row["End_Date"]
    district = row["District_Clean"]

    dates = pd.date_range(
        start=start,
        end=end,
        freq="D"
    )

    for date in dates:

        flood_days.append(
            {
                "District": district,
                "Date": date,
                "Flood_Observed": 1
            }
        )


flood_labels = pd.DataFrame(
    flood_days
)


# If duplicate events overlap,
# a day should still only have one flood label.

if len(flood_labels) > 0:

    flood_labels = (
        flood_labels
        .groupby(
            ["District", "Date"],
            as_index=False
        )["Flood_Observed"]
        .max()
    )

else:

    flood_labels = pd.DataFrame(
        columns=[
            "District",
            "Date",
            "Flood_Observed"
        ]
    )


print(
    "Unique flood-labelled days:",
    len(flood_labels)
)


# ============================================================
# 11. MERGE FLOOD LABELS WITH RAINFALL
# ============================================================

daily = daily.merge(
    flood_labels,
    on=["District", "Date"],
    how="left"
)

daily["Flood_Observed"] = (
    daily["Flood_Observed"]
    .fillna(0)
    .astype(int)
)


# Compatibility label
daily["Flood"] = daily["Flood_Observed"]


# ============================================================
# 12. CREATE 24-HOUR TARGET
# ============================================================

daily = daily.sort_values(
    ["District", "Date"]
).reset_index(drop=True)


daily["Flood_24h"] = (
    daily
    .groupby("District")["Flood_Observed"]
    .shift(-1)
)


# Remove final day of every district because
# there is no next-day observation.

daily = daily[
    daily["Flood_24h"].notna()
].copy()

daily["Flood_24h"] = (
    daily["Flood_24h"]
    .astype(int)
)


# ============================================================
# 13. DISTRICT CODE
# ============================================================

daily["District_Code"] = (
    daily["District"]
    .map(DISTRICT_CODE)
    .astype(int)
)


# ============================================================
# 14. FINAL COLUMN ORDER
# ============================================================

final_columns = [
    "Date",
    "Year",
    "Month",
    "DayOfYear",
    "District",
    "District_Code",

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

    "Flood",
    "Flood_Observed",
    "Flood_24h",
]


daily = daily[final_columns]


# ============================================================
# 15. SAVE DATASET
# ============================================================

os.makedirs(
    os.path.dirname(OUTPUT_FILE),
    exist_ok=True
)

daily.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# 16. VALIDATION SUMMARY
# ============================================================

print()
print("=" * 70)
print("FINAL DATASET SUMMARY")
print("=" * 70)

print("Rows:", len(daily))

print(
    "Date range:",
    daily["Date"].min(),
    "to",
    daily["Date"].max()
)

print()
print("District distribution:")

print(
    daily["District"]
    .value_counts()
    .sort_index()
)


print()
print("Flood_Observed distribution:")

print(
    daily["Flood_Observed"]
    .value_counts()
    .sort_index()
)


print()
print("Flood_24h distribution:")

print(
    daily["Flood_24h"]
    .value_counts()
    .sort_index()
)


print()
print("Flood_24h by district:")

print(
    daily
    .groupby("District")["Flood_24h"]
    .sum()
    .sort_index()
)


print()
print("Flood_24h by year:")

print(
    daily
    .groupby("Year")["Flood_24h"]
    .sum()
    .to_string()
)


print()
print("Saved successfully:")
print(OUTPUT_FILE)

print("=" * 70)