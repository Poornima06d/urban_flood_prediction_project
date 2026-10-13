import os
import joblib
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report
)

# ============================================================
# CONFIGURATION
# ============================================================

DATA_PATH = "data/processed/flood_training_24h.csv"
MODEL_PATH = "model/xgboost_flood_24h_model_v5.pkl"

TRAIN_END_YEAR = 2014
VALIDATION_START_YEAR = 2015
VALIDATION_END_YEAR = 2018
TEST_START_YEAR = 2019

RANDOM_STATE = 42


# ============================================================
# BASE FEATURES
# ============================================================

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
    "District_Code"
]

NORMALIZED_FEATURES = [
    "Rainfall_ratio_district",
    "Rain_lag1_ratio_district",
    "Rain_7d_ratio_district",
    "Rain_30d_ratio_district",
    "Rainfall_zscore_district",
    "Rain_7d_accum_ratio_district",
    "Rain_30d_accum_ratio_district"
]

FEATURES = BASE_FEATURES + NORMALIZED_FEATURES


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 75)
print("V5 - XGBOOST FLOOD PREDICTION")
print("TRAIN / VALIDATION / FINAL TEST")
print("=" * 75)

df = pd.read_csv(DATA_PATH)

df["Date"] = pd.to_datetime(df["Date"])

df = df.sort_values(
    ["District", "Date"]
).reset_index(drop=True)

print("\nDataset shape:", df.shape)
print(
    "Date range:",
    df["Date"].min().date(),
    "to",
    df["Date"].max().date()
)


# ============================================================
# CHRONOLOGICAL SPLIT
# ============================================================

train_df = df[
    df["Date"].dt.year <= TRAIN_END_YEAR
].copy()

validation_df = df[
    (df["Date"].dt.year >= VALIDATION_START_YEAR) &
    (df["Date"].dt.year <= VALIDATION_END_YEAR)
].copy()

test_df = df[
    df["Date"].dt.year >= TEST_START_YEAR
].copy()


print("\nDATA SPLIT")
print("-" * 50)

print(
    "Training:",
    train_df["Date"].min().date(),
    "to",
    train_df["Date"].max().date(),
    "| Rows:",
    len(train_df)
)

print(
    "Validation:",
    validation_df["Date"].min().date(),
    "to",
    validation_df["Date"].max().date(),
    "| Rows:",
    len(validation_df)
)

print(
    "Final Test:",
    test_df["Date"].min().date(),
    "to",
    test_df["Date"].max().date(),
    "| Rows:",
    len(test_df)
)


# ============================================================
# DISTRICT BASELINES
#
# IMPORTANT:
# Calculate only from TRAINING data.
# ============================================================

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

print("\nDISTRICT BASELINES")
print("-" * 50)
print(baseline.to_string(index=False))


# ============================================================
# MERGE BASELINES
# ============================================================

train_df = train_df.merge(
    baseline,
    on="District",
    how="left"
)

validation_df = validation_df.merge(
    baseline,
    on="District",
    how="left"
)

test_df = test_df.merge(
    baseline,
    on="District",
    how="left"
)


# ============================================================
# CREATE DISTRICT-NORMALIZED FEATURES
# ============================================================

def create_normalized_features(data):

    data = data.copy()

    data["Rainfall_ratio_district"] = (
        data["Rainfall_mm"] /
        data["Rain_mean"]
    )

    data["Rain_lag1_ratio_district"] = (
        data["Rain_lag1"] /
        data["Rain_mean"]
    )

    data["Rain_7d_ratio_district"] = (
        data["Rain_7d_avg_prev"] /
        data["Rain_mean"]
    )

    data["Rain_30d_ratio_district"] = (
        data["Rain_30d_avg_prev"] /
        data["Rain_mean"]
    )

    data["Rainfall_zscore_district"] = (
        (data["Rainfall_mm"] - data["Rain_mean"]) /
        data["Rain_std"]
    )

    data["Rain_7d_accum_ratio_district"] = (
        data["Rain_7d_prev"] /
        (data["Rain_mean"] * 7)
    )

    data["Rain_30d_accum_ratio_district"] = (
        data["Rain_30d_prev"] /
        (data["Rain_mean"] * 30)
    )

    data.replace(
        [np.inf, -np.inf],
        np.nan,
        inplace=True
    )

    data.fillna(0, inplace=True)

    return data


train_df = create_normalized_features(train_df)
validation_df = create_normalized_features(validation_df)
test_df = create_normalized_features(test_df)


# ============================================================
# PREPARE DATA
# ============================================================

X_train = train_df[FEATURES]
y_train = train_df["Flood_24h"]

X_validation = validation_df[FEATURES]
y_validation = validation_df["Flood_24h"]

X_test = test_df[FEATURES]
y_test = test_df["Flood_24h"]


print("\nFEATURE INFORMATION")
print("-" * 50)
print("Number of features:", len(FEATURES))

print("\nTraining flood distribution:")
print(y_train.value_counts())

print("\nValidation flood distribution:")
print(y_validation.value_counts())

print("\nFinal test flood distribution:")
print(y_test.value_counts())


# ============================================================
# CLASS IMBALANCE
# ============================================================

negative = (y_train == 0).sum()
positive = (y_train == 1).sum()

scale_pos_weight = negative / positive

print(
    "\nscale_pos_weight:",
    round(scale_pos_weight, 4)
)


# ============================================================
# XGBOOST MODEL
# ============================================================

model = XGBClassifier(
    n_estimators=600,
    max_depth=4,
    learning_rate=0.035,
    min_child_weight=5,
    subsample=0.85,
    colsample_bytree=0.85,
    gamma=0.15,
    reg_alpha=0.2,
    reg_lambda=2.0,

    objective="binary:logistic",
    eval_metric="logloss",

    scale_pos_weight=scale_pos_weight,

    random_state=RANDOM_STATE,
    n_jobs=4
)


# ============================================================
# TRAIN
# ============================================================

print("\nTraining V5 model...")

model.fit(
    X_train,
    y_train,
    eval_set=[(X_validation, y_validation)],
    verbose=False
)

print("Training completed.")


# ============================================================
# VALIDATION PREDICTIONS
# ============================================================

validation_probability = (
    model.predict_proba(X_validation)[:, 1]
)


# ============================================================
# SELECT THRESHOLD USING VALIDATION ONLY
# ============================================================

thresholds = np.arange(
    0.10,
    0.51,
    0.05
)

validation_results = []

for threshold in thresholds:

    validation_prediction = (
        validation_probability >= threshold
    ).astype(int)

    precision = precision_score(
        y_validation,
        validation_prediction,
        zero_division=0
    )

    recall = recall_score(
        y_validation,
        validation_prediction,
        zero_division=0
    )

    f1 = f1_score(
        y_validation,
        validation_prediction,
        zero_division=0
    )

    accuracy = accuracy_score(
        y_validation,
        validation_prediction
    )

    validation_results.append({
        "Threshold": round(threshold, 2),
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "Predicted_Floods": int(
            validation_prediction.sum()
        )
    })


validation_results_df = pd.DataFrame(
    validation_results
)

print("\nVALIDATION THRESHOLD RESULTS")
print("-" * 75)

print(
    validation_results_df
    .round(4)
    .to_string(index=False)
)


# ============================================================
# SELECT BEST VALIDATION THRESHOLD
# ============================================================

eligible = validation_results_df[
    validation_results_df["Recall"] >= 0.30
]

if len(eligible) > 0:

    best_row = eligible.loc[
        eligible["F1"].idxmax()
    ]

else:

    best_row = validation_results_df.loc[
        validation_results_df["F1"].idxmax()
    ]

BEST_THRESHOLD = float(
    best_row["Threshold"]
)

print(
    "\nSelected threshold from VALIDATION:",
    BEST_THRESHOLD
)


# ============================================================
# FINAL TEST PREDICTION
# ============================================================

test_probability = (
    model.predict_proba(X_test)[:, 1]
)

test_prediction = (
    test_probability >= BEST_THRESHOLD
).astype(int)


# ============================================================
# FINAL TEST METRICS
# ============================================================

accuracy = accuracy_score(
    y_test,
    test_prediction
)

precision = precision_score(
    y_test,
    test_prediction,
    zero_division=0
)

recall = recall_score(
    y_test,
    test_prediction,
    zero_division=0
)

f1 = f1_score(
    y_test,
    test_prediction,
    zero_division=0
)

roc_auc = roc_auc_score(
    y_test,
    test_probability
)

pr_auc = average_precision_score(
    y_test,
    test_probability
)

cm = confusion_matrix(
    y_test,
    test_prediction
)


print("\n")
print("=" * 75)
print("V5 FINAL TEST RESULTS")
print("=" * 75)

print("Threshold:", BEST_THRESHOLD)
print("Accuracy :", round(accuracy, 4))
print("Precision:", round(precision, 4))
print("Recall   :", round(recall, 4))
print("F1 Score :", round(f1, 4))
print("ROC-AUC  :", round(roc_auc, 4))
print("PR-AUC   :", round(pr_auc, 4))

print("\nConfusion Matrix:")
print(cm)

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        test_prediction,
        target_names=[
            "No Flood",
            "Flood"
        ],
        zero_division=0
    )
)


# ============================================================
# RISK CLASSIFICATION
# ============================================================

def classify_risk(probability):

    if probability < BEST_THRESHOLD:
        return "Low"

    elif probability < 0.70:
        return "Medium"

    else:
        return "High"


risk_level = [
    classify_risk(p)
    for p in test_probability
]


# ============================================================
# SAVE FINAL TEST PREDICTIONS
# ============================================================

prediction_output = test_df[
    [
        "Date",
        "District",
        "Flood_24h"
    ]
].copy()

prediction_output[
    "Flood_Probability"
] = test_probability

prediction_output[
    "Prediction"
] = test_prediction

prediction_output[
    "Risk_Level"
] = risk_level

prediction_output.to_csv(
    "model/test_predictions_v5.csv",
    index=False
)


# ============================================================
# SAVE VALIDATION THRESHOLDS
# ============================================================

validation_results_df.to_csv(
    "model/validation_threshold_results_v5.csv",
    index=False
)


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

importance_df = pd.DataFrame({
    "Feature": FEATURES,
    "Importance": model.feature_importances_
})

importance_df = importance_df.sort_values(
    "Importance",
    ascending=False
)

importance_df.to_csv(
    "model/feature_importance_v5.csv",
    index=False
)


print("\nTOP FEATURES")
print("-" * 50)

print(
    importance_df
    .head(15)
    .to_string(index=False)
)


# ============================================================
# SAVE MODEL
# ============================================================

joblib.dump(
    model,
    MODEL_PATH
)


# ============================================================
# SAVE METADATA
# ============================================================

metadata_path = (
    "model/model_metadata_v5.txt"
)

with open(metadata_path, "w") as f:

    f.write(
        "Urban Flood Prediction - XGBoost V5\n"
    )

    f.write("=" * 60 + "\n\n")

    f.write(
        "Training period: 1967-2014\n"
    )

    f.write(
        "Validation period: 2015-2018\n"
    )

    f.write(
        "Final test period: 2019-2023\n\n"
    )

    f.write(
        f"Number of features: {len(FEATURES)}\n"
    )

    f.write(
        f"Selected threshold: {BEST_THRESHOLD}\n\n"
    )

    f.write(
        f"Accuracy: {accuracy:.4f}\n"
    )

    f.write(
        f"Precision: {precision:.4f}\n"
    )

    f.write(
        f"Recall: {recall:.4f}\n"
    )

    f.write(
        f"F1 Score: {f1:.4f}\n"
    )

    f.write(
        f"ROC-AUC: {roc_auc:.4f}\n"
    )

    f.write(
        f"PR-AUC: {pr_auc:.4f}\n\n"
    )

    f.write(
        "District-normalized rainfall features:\n"
    )

    for feature in NORMALIZED_FEATURES:
        f.write(f"- {feature}\n")


# ============================================================
# COMPLETION
# ============================================================

print("\nFiles saved:")
print(
    "model/xgboost_flood_24h_model_v5.pkl"
)
print(
    "model/test_predictions_v5.csv"
)
print(
    "model/validation_threshold_results_v5.csv"
)
print(
    "model/feature_importance_v5.csv"
)
print(
    "model/model_metadata_v5.txt"
)

print("\nV5 training completed successfully.")