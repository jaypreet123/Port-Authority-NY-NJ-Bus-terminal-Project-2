#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# ============================================================
# GOAL 1: Top 5 Factors Affecting Bridge/Terminal Usage
# ============================================================

import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNet
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt

# ---------------- CONFIG ----------------
FILE = "/Users/jay/Downloads/Goal1_Consolidated_Traffic.csv"
OUT_DIR = "/Users/jay/Downloads/Goal1_Results"
os.makedirs(OUT_DIR, exist_ok=True)

# ---------------- LOAD ----------------
df = pd.read_csv(FILE)
print(f"Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")

# ---------------- CLEAN / ENCODE ----------------
# Convert DATE column to datetime and extract useful fields
if "DATE" in df.columns:
    df["DATE"] = pd.to_datetime(df["DATE"])
    df["Year"] = df["DATE"].dt.year
    df["Month"] = df["DATE"].dt.month
    df["DayOfWeek"] = df["DATE"].dt.dayofweek
    df.drop(columns=["DATE"], inplace=True)

# Remove leakage columns (Autos + Buses + Trucks = Total Traffic)
leak_cols = ["AUTOS", "BUSES", "SMALL_T", "LARGE_T"]
if all(c in df.columns for c in leak_cols):
    if np.allclose(df[leak_cols].sum(axis=1), df["Total_Traffic"], rtol=0.01):
        df.drop(columns=leak_cols, inplace=True)
        print("Leakage columns removed:", leak_cols)

# Encode categorical columns with one-hot encoding
cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
if cat_cols:
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)
    print("Categorical columns encoded:", cat_cols)

# ---------------- TRAIN MODEL ----------------
X = df.drop(columns=["Total_Traffic"])
y = df["Total_Traffic"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scale features before regression
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

# ElasticNet regression model (tuned parameters)
model = ElasticNet(alpha=0.1061579, l1_ratio=0.01, random_state=42, max_iter=10000)
model.fit(X_train_s, y_train)

# ---------------- METRICS ----------------
y_pred = model.predict(X_test_s)

r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)

print(f"\nModel Performance:\nR² = {r2:.4f}, RMSE = {rmse:.2f}, MAE = {mae:.2f}")

# ---------------- FEATURE IMPORTANCE ----------------
coef_df = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": model.coef_,
    "AbsCoeff": np.abs(model.coef_)
}).sort_values("AbsCoeff", ascending=False)

top5 = coef_df.head(5)

print("\nTop 5 Factors Affecting Bridge Usage:")
for _, row in top5.iterrows():
    direction = "increases" if row.Coefficient > 0 else "decreases"
    print(f"  {row.Feature}: {direction} traffic ({row.Coefficient:.4f})")

# ---------------- VISUALIZE ----------------
plt.figure(figsize=(8, 5))

colors = ["green" if c > 0 else "red" for c in top5["Coefficient"]]
plt.barh(top5["Feature"], top5["Coefficient"], color=colors)

plt.xlabel("Coefficient Value (Impact on Traffic)")
plt.title("Top 5 Factors Affecting Bridge/Terminal Usage")
plt.axvline(0, color="black", linewidth=1)
plt.tight_layout()

plot_path = os.path.join(OUT_DIR, "top5_factors.png")
plt.savefig(plot_path, dpi=300)

print(f"\nSaved: {plot_path}")

# ---------------- SAVE RESULTS ----------------
top5.to_csv(os.path.join(OUT_DIR, "top5_factors.csv"), index=False)

metrics_path = os.path.join(OUT_DIR, "model_metrics.csv")
pd.DataFrame({"R2": [r2], "RMSE": [rmse], "MAE": [mae]}).to_csv(metrics_path, index=False)

print("Goal 1 Files", OUT_DIR)

