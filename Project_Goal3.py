#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# ==============================
# Goal 3: Busiest Times & Factors + Forecasting 
# Daily Aggregated + Weather + Speed + Holidays, Forecast Until 2030
# ==============================

import pandas as pd
import os
from prophet import Prophet
import matplotlib.pyplot as plt

# === Step 1: File paths ===
input_dir = "/Users/jay/Desktop/Port Authority Project 2_Tunnels and Bridges/Final Dataset/Traffic_By_Facility"
weather_file = "/Users/jay/Desktop/Port Authority Project 2_Tunnels and Bridges/Final Dataset/Tbl_Weather_clean.csv"
holiday_file = "/Users/jay/Desktop/Port Authority Project 2_Tunnels and Bridges/Final Dataset/US_Holiday_Dates_clean.csv"
speed_file = "/Users/jay/Desktop/Port Authority Project 2_Tunnels and Bridges/Final Dataset/Facility_Mobility_Speeds_clean.csv"
output_dir = "/Users/jay/Desktop/Traffic_Goal3"

os.makedirs(output_dir, exist_ok=True)

# === Step 2: Load supporting datasets ===
weather = pd.read_csv(weather_file)
holidays = pd.read_csv(holiday_file)
speed = pd.read_csv(speed_file)

# Keep only the relevant speed columns
speed = speed[['DATE', 'Freeflow_Speed', 'Avg_Speed', 'Delta']]

# Convert dates
weather['DATE'] = pd.to_datetime(weather['DATE'], errors='coerce')
holidays['DATE'] = pd.to_datetime(holidays['DATE'], errors='coerce')
speed['DATE'] = pd.to_datetime(speed['DATE'], errors='coerce')

# === Step 3: Container to store results for all facilities ===
all_facilities_data = []

# === Step 4: Process each facility ===
facility_files = [f for f in os.listdir(input_dir) if f.endswith("_traffic.csv")]

for file in facility_files:
    facility_name = file.replace("_traffic.csv", "")
    print(f"\nProcessing {facility_name}...")

    # Load data
    traffic = pd.read_csv(os.path.join(input_dir, file))
    traffic['DATE'] = pd.to_datetime(traffic['DATE'], errors='coerce')

    print(f"Data coverage before fixing dates: {traffic['DATE'].min().date()} → {traffic['DATE'].max().date()}")

    # Ensure continuity in timeline up to May 31, 2025
    full_end_date = pd.to_datetime("2025-05-31")
    all_dates = pd.date_range(start=traffic['DATE'].min(), end=full_end_date, freq='D')

    traffic_full = pd.DataFrame({'DATE': all_dates})
    traffic_full = traffic_full.merge(traffic, on='DATE', how='left')
    traffic_full['Facility_Name'] = facility_name

    # Clean numeric columns
    numeric_cols = ['TOTAL', 'VIOLATION', 'Autos', 'Small_T', 'Large_T', 'Buses']
    for col in numeric_cols:
        if col in traffic_full.columns:
            traffic_full[col] = pd.to_numeric(traffic_full[col], errors='coerce')
            traffic_full[col] = traffic_full[col].fillna(method='ffill').fillna(traffic_full[col].mean())

    traffic = traffic_full.copy()

    print(f"Data coverage after fixing dates: {traffic['DATE'].min().date()} → {traffic['DATE'].max().date()}")

    # === Aggregate to daily level ===
    daily = (
        traffic.groupby(['DATE', 'Facility_Name'])
        .agg({
            'TOTAL': 'sum',
            'VIOLATION': 'sum',
            'Autos': 'sum',
            'Small_T': 'sum',
            'Large_T': 'sum',
            'Buses': 'sum'
        })
        .reset_index()
    )

    # Add calendar features
    daily['Year'] = daily['DATE'].dt.year
    daily['Month'] = daily['DATE'].dt.month
    daily['Week'] = daily['DATE'].dt.isocalendar().week
    daily['Day_Name'] = daily['DATE'].dt.day_name()

    # Add holiday information
    daily = daily.merge(holidays[['DATE', 'Holiday']], on='DATE', how='left')
    daily['Holiday_Flag'] = daily['Holiday'].notna().astype(int)

    # Merge with weather & speed data
    merged = daily.merge(weather, on='DATE', how='left')
    merged = merged.merge(speed, on='DATE', how='left')

    # Clean weather & speed columns
    weather_cols = [c for c in weather.columns if c != 'DATE']
    speed_cols = ['Freeflow_Speed', 'Avg_Speed', 'Delta']

    for col in weather_cols + speed_cols:
        if col in merged.columns:
            merged[col] = pd.to_numeric(merged[col], errors='coerce')
            merged[col] = merged.groupby('Month')[col].transform(lambda x: x.fillna(x.mean()))
            merged[col] = merged[col].fillna(merged[col].mean())

    # === Step 5: Forecasting (Prophet) ===
    regressors = [
        c for c in ['Holiday_Flag', 'TMAX', 'PRCP', 'Freeflow_Speed', 'Avg_Speed', 'Delta']
        if c in merged.columns
    ]

    df = merged[['DATE', 'TOTAL'] + regressors].rename(columns={'DATE': 'ds', 'TOTAL': 'y'})
    df = df.dropna(subset=['y'])

    # Replace incompatible values for Prophet
    df = df.replace([float("inf"), float("-inf")], None)
    df.columns = [c.replace(" ", "_").replace("-", "_") for c in df.columns]

    # Final clean of regressors
    for reg in regressors:
        if reg in df.columns:
            df[reg] = pd.to_numeric(df[reg], errors='coerce')
            df[reg] = df[reg].fillna(df[reg].mean())

    df = df.dropna(subset=regressors + ['y'])

    # Run Prophet only when dataset is large enough
    if df.shape[0] >= 30:

        m = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=False
        )

        for reg in regressors:
            m.add_regressor(reg)

        m.fit(df)

        # Forecast until December 31, 2030
        last_date = merged['DATE'].max()
        target_date = pd.to_datetime("2030-12-31")
        periods = (target_date - last_date).days
        future = m.make_future_dataframe(periods=periods)

        # Fill regressor values into the future
        for reg in regressors:
            if reg == 'Holiday_Flag':
                future[reg] = 0
            else:
                future[reg] = df[reg].mean()

        forecast = m.predict(future)
        forecast_part = forecast[forecast['ds'] > last_date].copy()
        forecast_part = forecast_part.rename(columns={'ds': 'DATE', 'yhat': 'TOTAL'})
        forecast_part = forecast_part[['DATE', 'TOTAL']]

        # Scale other columns proportionally
        ratios = {}
        for col in ['Autos', 'Small_T', 'Large_T', 'Buses', 'VIOLATION']:
            ratios[col] = (merged[col].sum() / merged['TOTAL'].sum()) if merged['TOTAL'].sum() > 0 else 0

        forecast_part['Facility_Name'] = facility_name
        forecast_part['VIOLATION'] = forecast_part['TOTAL'] * ratios['VIOLATION']
        forecast_part['Autos'] = forecast_part['TOTAL'] * ratios['Autos']
        forecast_part['Small_T'] = forecast_part['TOTAL'] * ratios['Small_T']
        forecast_part['Large_T'] = forecast_part['TOTAL'] * ratios['Large_T']
        forecast_part['Buses'] = forecast_part['TOTAL'] * ratios['Buses']

        # Add calendar fields for forecast
        forecast_part['Year'] = forecast_part['DATE'].dt.year
        forecast_part['Month'] = forecast_part['DATE'].dt.month
        forecast_part['Week'] = forecast_part['DATE'].dt.isocalendar().week
        forecast_part['Day_Name'] = forecast_part['DATE'].dt.day_name()

        # Add holiday flag to forecast
        forecast_part = forecast_part.merge(holidays[['DATE', 'Holiday']], on='DATE', how='left')
        forecast_part['Holiday_Flag'] = forecast_part['Holiday'].notna().astype(int)

        # Fill missing weather/speed with mean
        for col in weather_cols + speed_cols:
            if col not in forecast_part.columns:
                forecast_part[col] = merged[col].mean()

        # Arrange columns to match historical dataset
        forecast_part = forecast_part[merged.columns.tolist()]

        # Combine historical + forecast
        merged = pd.concat([merged, forecast_part], ignore_index=True)

        # Save forecast plot
        fig = m.plot(forecast)
        plt.title(f"{facility_name} - Daily Traffic Forecast (Until Dec 2030)")
        plot_file = os.path.join(output_dir, f"{facility_name}_Goal3_Forecast.png")
        fig.savefig(plot_file)
        plt.close(fig)

    else:
        print(f"Skipping forecasting for {facility_name}: Not enough data.")

    # === Step 6: Save per-facility dataset ===
    out_file = os.path.join(output_dir, f"{facility_name}_Goal3_Traffic_Factors.csv")
    merged.to_csv(out_file, index=False)
    print(f"Saved dataset for {facility_name}: {out_file} (rows={merged.shape[0]})")

    all_facilities_data.append(merged)

# === Step 7: Save combined dataset ===
combined_df = pd.concat(all_facilities_data, ignore_index=True)
combined_out = os.path.join(output_dir, "All_Facilities_Goal3_Traffic_Factors.csv")
combined_df.to_csv(combined_out, index=False)

print("\nGoal 3 completed for all facilities.")
print(f"Combined dataset saved: {combined_out}")

