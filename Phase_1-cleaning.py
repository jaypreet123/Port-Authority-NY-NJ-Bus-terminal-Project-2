#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import os

# File paths
input_file = "/Users/jay/Desktop/Port Authority Project 2_Tunnels and Bridges/DataSet/All Recorded Traffic.txt"
output_dir = "/Users/jay/Desktop/Traffic_By_Facility"

# Create output folder
os.makedirs(output_dir, exist_ok=True)

# ---------------------------------------------------------
# STEP 1: Load the dataset
# ---------------------------------------------------------
try:
    traffic = pd.read_csv(input_file, sep="\t", engine="python")
except:
    traffic = pd.read_csv(input_file, sep=",", engine="python")

# ---------------------------------------------------------
# STEP 2: Clean the column names
# - Remove duplicated columns
# - Remove any "Unnamed" auto-generated columns
# ---------------------------------------------------------
traffic = traffic.loc[:, ~traffic.columns.duplicated()]
traffic = traffic[[col for col in traffic.columns if not col.strip().startswith("Unnamed")]]

# ---------------------------------------------------------
# STEP 3: Fix DATE column
# Convert to datetime.date and drop invalid dates
# ---------------------------------------------------------
if "DATE" in traffic.columns:
    traffic["DATE"] = pd.to_datetime(traffic["DATE"], errors="coerce").dt.date
    traffic = traffic.dropna(subset=["DATE"])

# ---------------------------------------------------------
# STEP 4: Fix TIME column
# Convert HHMM → proper time format
# ---------------------------------------------------------
if "TIME" in traffic.columns:
    traffic["TIME"] = pd.to_datetime(
        traffic["TIME"].astype(str).str.zfill(4),   # pad "830" → "0830"
        format="%H%M",
        errors="coerce"
    ).dt.time
    traffic = traffic.dropna(subset=["TIME"])

# ---------------------------------------------------------
# STEP 5: Map facility codes (FAC)
# ---------------------------------------------------------
fac_map = {
    1: "Holland",
    2: "Lincoln",
    4: "Bayonne",
    5: "Goethals",
    6: "Outerbridge",
    7: "GWB_Upper",
    8: "GWB_PIP",
    9: "GWB_Lower"
}

traffic["FAC"] = pd.to_numeric(traffic["FAC"], errors="coerce")
traffic["Facility_Name"] = traffic["FAC"].map(fac_map)

# ---------------------------------------------------------
# STEP 6: Aggregate by Facility + Date + Time
# - Sum numeric columns
# - Keep the first value for non-numeric columns
# ---------------------------------------------------------
agg_dict = {}
for col in traffic.columns:
    if pd.api.types.is_numeric_dtype(traffic[col]):
        agg_dict[col] = "sum"
    else:
        agg_dict[col] = "first"

traffic_grouped = traffic.groupby(
    ["Facility_Name", "DATE", "TIME"], as_index=False
).agg(agg_dict)

# ---------------------------------------------------------
# STEP 7: Save each facility into its own CSV file
# ---------------------------------------------------------
for fac_name in traffic_grouped["Facility_Name"].unique():
    subset = traffic_grouped[traffic_grouped["Facility_Name"] == fac_name]

    if not subset.empty:
        file_path = os.path.join(output_dir, f"{fac_name}_traffic.csv")
        subset.to_csv(file_path, index=False)

        print(
            f"Saved {fac_name} data to {file_path} "
            f"sample DATEs: {subset['DATE'].head(3).tolist()} "
            f"sample TIMEs: {subset['TIME'].head(3).tolist()}"
        )
    else:
        print(f"No data found for {fac_name}")

