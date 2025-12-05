#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# ==============================
# Goal 2: Toll Violations Workflow for All Facilities
# ==============================

import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from xgboost import XGBClassifier
import matplotlib.pyplot as plt

# === Step 1: File paths for all facilities ===
facility_files = {
    "Bayonne": "Bayonne_traffic.csv",
    "GWB_Lower": "GWB_Lower_traffic.csv",
    "GWB_PIP": "GWB_PIP_traffic.csv",
    "GWB_Upper": "GWB_Upper_traffic.csv",
    "Goethals": "Goethals_traffic.csv",
    "Holland": "Holland_traffic.csv",
    "Lincoln": "Lincoln_traffic.csv",
    "Outerbridge": "Outerbridge_traffic.csv"
}

input_dir = "/Users/jay/Desktop/Port Authority Project 2_Tunnels and Bridges/Final Dataset/Traffic_By_Facility"
output_dir = "/Users/jay/Desktop/Traffic_Goal2"
os.makedirs(output_dir, exist_ok=True)

# === Step 2: Process each facility ===
all_facility_data = []  # store outputs for final combined export

for facility, filename in facility_files.items():
    print(f"\nProcessing {facility}...")

    # Load the dataset safely
    filepath = os.path.join(input_dir, filename)

    try:
        traffic = pd.read_csv(filepath)
    except Exception as e:
        print(f"Error loading {filename}: {e}")
        continue

    # Fix DATE column
    traffic['DATE'] = pd.to_datetime(traffic['DATE'], errors='coerce')

    # Extract useful time-based features
    traffic['Year'] = traffic['DATE'].dt.year
    traffic['Month'] = traffic['DATE'].dt.month
    traffic['Week'] = traffic['DATE'].dt.isocalendar().week
    traffic['DayOfWeek'] = traffic['DATE'].dt.day_name()

    # Aggregate to daily-level dataset for ML
    violations_dataset = (
        traffic.groupby(['DATE', 'Year', 'Month', 'Week', 'Facility_Name'])
        .agg(
            Total_Traffic=('TOTAL', 'sum'),
            Cash=('CASH', 'sum'),
            EZPass=('EZPASS', 'sum'),
            Violations=('VIOLATION', 'sum')
        )
        .reset_index()
    )

    # Create violation rate
    violations_dataset['Violation_Rate'] = (
        violations_dataset['Violations'] /
        violations_dataset['Total_Traffic'].replace(0, pd.NA)
    )

    # Save BI dataset
    out_csv = os.path.join(output_dir, f"{facility}_Violations_Dataset.csv")
    try:
        violations_dataset.to_csv(out_csv, index=False)
    except Exception as e:
        print(f"Error saving BI dataset for {facility}: {e}")
        continue

    print(f"Saved BI dataset for {facility}: {out_csv}")

    # Keep for combined export
    violations_dataset['Facility'] = facility
    all_facility_data.append(violations_dataset)

    # ========================
    # Part 2: Predictive Models
    # ========================

    # ML target variable:
    # 1 = there was at least one violation on that day
    # 0 = no violations that day
    violations_dataset['Violation_Flag'] = (violations_dataset['Violations'] > 0).astype(int)

    # Feature set for the model
    features = ['Total_Traffic', 'Cash', 'EZPass', 'Violation_Rate', 'Year', 'Month', 'Week']
    X = violations_dataset[features].fillna(0)
    y = violations_dataset['Violation_Flag']

    # Skip modeling if there is only one class
    if len(y.unique()) < 2:
        print(f"Skipping modeling for {facility}: only one class present ({y.unique()[0]})")
        continue

    # Train-test split (stratified to maintain same class ratio in train/test)
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
    except Exception as e:
        print(f"Skipping modeling for {facility}: train-test split error: {e}")
        continue

    # ---------------------------------------------------
    # Logistic Regression Model
    # ---------------------------------------------------
    # This model estimates the probability of having violations
    # based on features like traffic volume, EZPass, cash, etc.
    try:
        log_reg = LogisticRegression(max_iter=1000)
        log_reg.fit(X_train, y_train)
        y_pred_log = log_reg.predict(X_test)
    except Exception as e:
        print(f"Logistic Regression failed for {facility}: {e}")
        continue

    print("\n   Logistic Regression Results:")
    print(classification_report(y_test, y_pred_log))

    # ROC-AUC measures how well the model separates violation vs non-violation days
    try:
        roc_score = roc_auc_score(y_test, log_reg.predict_proba(X_test)[:, 1])
        print("   ROC-AUC:", roc_score)
    except Exception as e:
        print(f"ROC-AUC calculation error for {facility}: {e}")

    # Save coefficients (feature influence)
    coef_out = os.path.join(output_dir, f"{facility}_Violations_Logistic_Results.csv")
    try:
        coefficients = pd.DataFrame({
            "Feature": X.columns,
            "Coefficient": log_reg.coef_[0]
        }).sort_values(by="Coefficient", ascending=False)
        coefficients.to_csv(coef_out, index=False)
    except Exception as e:
        print(f"Error saving logistic coefficients for {facility}: {e}")

    # ---------------------------------------------------
    # XGBoost Model
    # ---------------------------------------------------
    # XGBoost captures non-linear relationships and interactions between variables,
    # making it stronger for patterns like spikes in violations on specific weeks.
    try:
        xgb = XGBClassifier(eval_metric='logloss', random_state=42)
        xgb.fit(X_train, y_train)
        y_pred_xgb = xgb.predict(X_test)
    except Exception as e:
        print(f"XGBoost failed for {facility}: {e}")
        continue

    print("\n   XGBoost Results:")
    print(classification_report(y_test, y_pred_xgb))

    try:
        roc_xgb = roc_auc_score(y_test, xgb.predict_proba(X_test)[:, 1])
        print("   ROC-AUC:", roc_xgb)
    except Exception as e:
        print(f"XGBoost ROC-AUC error for {facility}: {e}")

    # Save feature importances
    imp_out = os.path.join(output_dir, f"{facility}_Violations_XGBoost_Results.csv")
    try:
        importances = pd.DataFrame({
            "Feature": X.columns,
            "Importance": xgb.feature_importances_
        }).sort_values(by="Importance", ascending=False)
        importances.to_csv(imp_out, index=False)
    except Exception as e:
        print(f"Error saving XGBoost results for {facility}: {e}")

    # Save importance plot
    try:
        plt.figure(figsize=(8, 5))
        plt.barh(importances['Feature'], importances['Importance'])
        plt.xlabel("Importance Score")
        plt.title(f"XGBoost Feature Importances - {facility}")
        plt.gca().invert_yaxis()
        plt.tight_layout()

        plot_out = os.path.join(output_dir, f"{facility}_Violations_XGBoost_FeatureImportances.png")
        plt.savefig(plot_out)
        plt.close()
    except Exception as e:
        print(f"Error saving feature importance plot for {facility}: {e}")

    print(f"Logistic and XGBoost results saved for {facility}")

# === Step 3: Save combined file ===
if all_facility_data:
    try:
        combined_df = pd.concat(all_facility_data, ignore_index=True)
        combined_path = os.path.join(output_dir, "All_Facilities_Violations_Combined.csv")
        combined_df.to_csv(combined_path, index=False)
        print(f"\nCombined dataset for all facilities saved: {combined_path}")
    except Exception as e:
        print(f"Error creating combined violations dataset: {e}")
else:
    print("\nNo facility data available to combine.")

print("\nGoal 2 completed for all facilities.")

