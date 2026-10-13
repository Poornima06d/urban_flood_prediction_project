# Urban Flood Prediction & Early Warning System

## Project
A Machine Learning Approach to Urban Flood Prediction and Early Warning System Using XGBoost.

## Monitored districts
- Belagavi
- Bagalkot
- Shivamogga
- Chikkamagaluru
- Uttara Kannada

## Main stack
Python 3, Flask, HTML/CSS/JavaScript, MySQL, Pandas, NumPy, Scikit-learn, Matplotlib, XGBoost, Weather API, Leaflet/OpenStreetMap.

## Important data rule
Put your prepared CSV at:

`data/raw/imd_training_dataset_prepared.csv`

The CSV should contain:
`Date, Year, Month, District, Rainfall_mm, Rain_lag1, Rain_3d_prev, Rain_7d_prev, Rain_15d_prev, Rain_max_3d_prev, Rain_max_7d_prev, Flood`

## 24-hour target
The training script shifts the historical `Flood` label by one day within each district:

Today's rainfall-related features -> tomorrow's flood status.

This is the intended 24-hour early-warning formulation.

## Setup
1. Create a Python virtual environment.
2. Install packages:
   `pip install -r requirements.txt`
3. Copy `.env.example` to `.env`.
4. Add the OpenWeather API key.
5. Create the MySQL database by running `database/schema.sql`.
6. Copy the prepared CSV into `data/raw/`.
7. Train the model:
   `python model/train_model.py`
8. Start the web application:
   `python app.py`
9. Open the local Flask URL shown in the terminal.

## Pages
Home
Dashboard
Weather
Flood Prediction
Alerts
Risk Map
Prediction History
Rainfall Analysis
How It Works

## Important modeling note
The initial Low/Medium/High probability cutoffs are configuration values. They should be reviewed after model evaluation/calibration rather than presented as scientifically fixed thresholds.

## Operational note
The current scaffold uses OpenWeather current weather as the operational weather source. For a rigorous 24-hour deployment model, forecast rainfall should be incorporated and the training features should be aligned to the same forecast horizon.
