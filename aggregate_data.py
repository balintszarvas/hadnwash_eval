import pandas as pd
import numpy as np
from pathlib import Path

# Intitial setup
BASE_PATH = Path("raw_data")
LABELS = ["One", "Two", "Three", "Four", "Five"]
SENSORS = {
    "Accelerometer.csv": "acc_",
    "Linear Accelerometer.csv": "linacc_",
    "Gyroscope.csv": "gyr_",
    "Magnetometer.csv": "mag_"
}
GRANULARITY = '100ms'
all_data = []

# Function to create single sensor file
def load_and_aggregate_sensor(file_path, prefix):
    df = pd.read_csv(file_path)
    if "Time (s)" not in df.columns:
        raise ValueError(f"'Time (s)' column not found in {file_path}")
    df['timestamp'] = pd.to_datetime('2025-01-06') + pd.to_timedelta(df['Time (s)'], unit='s')
    df = df.set_index('timestamp')
    df = df.drop(columns=["Time (s)"])
    df = df.rename(columns=lambda col: prefix + col.strip().split(" ")[0].lower())
    agg_df = df.resample(GRANULARITY).mean()
    return agg_df

# Loop through each label folder and process sensor files
for label in LABELS:
    print(f"Processing label: {label}")
    folder_path = BASE_PATH / label
    dfs = []

    for sensor_file, prefix in SENSORS.items():
        file_path = folder_path / sensor_file
        sensor_df = load_and_aggregate_sensor(file_path, prefix)
        dfs.append(sensor_df)

    # Merge all sensors on timestamp
    merged = pd.concat(dfs, axis=1)

    # Add label columns (one-hot encoded)
    for lbl in LABELS:
        merged[f"label{lbl}"] = 1 if lbl == label else 0

    all_data.append(merged)

# Combine and reset index
final_df = pd.concat(all_data).reset_index()
final_df = final_df.rename(columns={'timestamp': 'datetime'})

# Save to csv file
final_df.to_csv("handwash_eval_result.csv", index=False)
print("Dataset succesfully saved")
