#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# ==============================
# Goal 5: Forecast Facility Usage (Year + Month)
# Actuals + Forecast to 2030
# Prophet vs SARIMAX Comparison + Validation
# ==============================

import pandas as pd
import os
from prophet import Prophet
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
import warnings
warnings.filterwarnings("ignore")

# === Step 1: File paths for all facilities ===
facility_files = {
    "Bayonne": "Bayonne_traffic.csv",
    "GWB_Lower": "GWB_Lower_traffic.csv",
    "GWB_PIP": "GWB_PIP_traffic.csv",  # only until 2022
    "GWB_Upper": "GWB_Upper_traffic.csv",
    "Goethals": "Goethals_traffic.csv",
    "Holland": "Holland_traffic.csv",
    "Lincoln": "Lincoln_traffic.csv",
    "Outerbridge": "Outerbridge_traffic.csv"
}

input_dir = "/Users/jay/Desktop/Port Authority Project 2_Tunnels and Bridges/Final Dataset/Traffic_By_Facility"
output_dir = "/Users/jay/Desktop/Traffic_Goal5"
os.makedirs(output_dir, exist_ok=True)

all_facilities_data = []

# === Step 2: Process each facility ===
for facility, filename in facility_files.items():
    print(f"\nProcessing {facility}...")

    filepath = os.path.join(input_dir, filename)
    df = pd.read_csv(filepath)

    # Fix date format
    df['DATE'] = pd.to_datetime(df['DATE'], errors='coerce')

    # Aggregate daily totals
    daily = (
        df.groupby('DATE')
        .agg({
            'TOTAL': 'sum',
            'Autos': 'sum',
            'Small_T': 'sum',
            'Large_T': 'sum',
            'Buses': 'sum',
            'VIOLATION': 'sum'
        })
        .reset_index()
    )

    # Ratios used later to distribute forecasted TOTAL into categories
    ratios = {}
    for col in ['Autos', 'Small_T', 'Large_T', 'Buses', 'VIOLATION']:
        ratios[col] = daily[col].sum() / daily['TOTAL'].sum() if daily['TOTAL'].sum() > 0 else 0

    # Prepare Prophet dataset
    prophet_df = daily[['DATE', 'TOTAL']].rename(columns={'DATE': 'ds', 'TOTAL': 'y'})

    # Skip facilities with very little data
    if prophet_df.shape[0] < 30:
        print(f"Skipping {facility}: not enough data points.")
        continue

    # === Step 2A: Train/Test Split for Model Validation ===
    cutoff_date = prophet_df['ds'].max() - pd.Timedelta(days=180)
    train_df = prophet_df[prophet_df['ds'] <= cutoff_date]
    test_df = prophet_df[prophet_df['ds'] > cutoff_date]

    # --------------------
    # Prophet Model
    # --------------------
    m = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=False,
        daily_seasonality=False
    )
    m.fit(train_df)

    future_test = test_df[['ds']]
    forecast_test = m.predict(future_test)

    prophet_pred = forecast_test['yhat'].values
    prophet_mae = mean_absolute_error(test_df['y'], prophet_pred)
    prophet_rmse = np.sqrt(mean_squared_error(test_df['y'], prophet_pred))

    # --------------------
    # SARIMAX Model
    # --------------------
    try:
        sarimax_train = train_df.set_index('ds')['y'].asfreq('D').fillna(method='ffill')
        sarimax_test = test_df.set_index('ds')['y'].asfreq('D').fillna(method='ffill')

        sarimax_model = SARIMAX(
            sarimax_train,
            order=(1, 1, 1),
            seasonal_order=(1, 1, 1, 7)
        )
        sarimax_fit = sarimax_model.fit(disp=False)

        sarimax_pred = sarimax_fit.forecast(steps=len(sarimax_test))

        sarimax_mae = mean_absolute_error(sarimax_test, sarimax_pred)
        sarimax_rmse = np.sqrt(mean_squared_error(sarimax_test, sarimax_pred))

    except Exception as e:
        sarimax_mae, sarimax_rmse = np.inf, np.inf
        print(f"SARIMAX failed for {facility}: {e}")

    # Compare validation results
    print(f"Prophet RMSE: {prophet_rmse:.2f}, SARIMAX RMSE: {sarimax_rmse:.2f}")
    best_model = "Prophet" if prophet_rmse <= sarimax_rmse else "SARIMAX"
    print(f"Best model selected for {facility}: {best_model}")

    # === Step 3: Train the Best Model on the Full Dataset ===
    last_date = prophet_df['ds'].max()

    if facility == "GWB_PIP":
        forecast_start = pd.to_datetime("2023-01-01")  # dataset shorter
    else:
        forecast_start = last_date + pd.Timedelta(days=1)

    target_date = pd.to_datetime("2030-12-31")
    periods = (target_date - forecast_start).days

    if best_model == "Prophet":
        m_full = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
        m_full.fit(prophet_df)

        future = m_full.make_future_dataframe(periods=periods, freq="D")
        forecast = m_full.predict(future)

        forecast_daily = forecast[['ds', 'yhat']].rename(columns={'ds': 'DATE', 'yhat': 'Forecast_TOTAL'})

    else:
        sarimax_full = prophet_df.set_index('ds')['y'].asfreq('D').fillna(method='ffill')

        sarimax_model_full = SARIMAX(
            sarimax_full,
            order=(1, 1, 1),
            seasonal_order=(1, 1, 1, 7)
        )
        sarimax_fit_full = sarimax_model_full.fit(disp=False)

        forecast_vals = sarimax_fit_full.forecast(steps=periods)

        forecast_daily = pd.DataFrame({
            'DATE': pd.date_range(forecast_start, target_date, freq='D'),
            'Forecast_TOTAL': forecast_vals.values
        })

    # Keep only forecasted rows (not historical)
    forecast_daily = forecast_daily[forecast_daily['DATE'] >= forecast_start]

    # Split TOTAL forecasts into categories using historical ratios
    for col in ['Autos', 'Small_T', 'Large_T', 'Buses', 'VIOLATION']:
        forecast_daily[f"Forecast_{col}"] = forecast_daily['Forecast_TOTAL'] * ratios[col]

    forecast_daily['Facility_Name'] = facility
    forecast_daily['Year'] = forecast_daily['DATE'].dt.year
    forecast_daily['Month'] = forecast_daily['DATE'].dt.month

    # === Step 4: Aggregate actuals to monthly totals ===
    actuals = (
        daily.copy()
        .assign(
            Facility_Name=facility,
            Year=daily['DATE'].dt.year,
            Month=daily['DATE'].dt.month
        )
        .groupby(['Facility_Name', 'Year', 'Month'])
        .agg({
            'TOTAL': 'sum',
            'Autos': 'sum',
            'Small_T': 'sum',
            'Large_T': 'sum',
            'Buses': 'sum',
            'VIOLATION': 'sum'
        })
        .reset_index()
    )

    # Rename for consistent combined output
    actuals = actuals.rename(columns={
        'TOTAL': 'Forecast_TOTAL',
        'Autos': 'Forecast_Autos',
        'Small_T': 'Forecast_Small_T',
        'Large_T': 'Forecast_Large_T',
        'Buses': 'Forecast_Buses',
        'VIOLATION': 'Forecast_VIOLATION'
    })
    actuals['Source'] = "Actual"

    # === Step 5: Aggregate forecast to monthly totals ===
    forecast_monthly = (
        forecast_daily.groupby(['Facility_Name', 'Year', 'Month'])
        .agg({
            'Forecast_TOTAL': 'sum',
            'Forecast_Autos': 'sum',
            'Forecast_Small_T': 'sum',
            'Forecast_Large_T': 'sum',
            'Forecast_Buses': 'sum',
            'Forecast_VIOLATION': 'sum'
        })
        .reset_index()
    )

    forecast_monthly['Source'] = f"Forecast_{best_model}"

    # === Step 6: Combine Actual + Forecast ===
    combined = pd.concat([actuals, forecast_monthly], ignore_index=True)

    # Save per-facility results
    facility_out = os.path.join(output_dir, f"{facility}_Forecast.csv")
    combined.to_csv(facility_out, index=False)
    print(f"Saved: {facility_out}")

    all_facilities_data.append(combined)

# === Step 7: Combine All Facilities into One File ===
final_combined = pd.concat(all_facilities_data, ignore_index=True)
combined_out = os.path.join(output_dir, "All_Facilities_Forecast.csv")
final_combined.to_csv(combined_out, index=False)

print("Prophet vs SARIMAX model comparison done and Forecasted till 2030.")

