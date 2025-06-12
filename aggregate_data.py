import pandas as pd
import numpy as np
from pathlib import Path

# Initial setup
BASE_PATH = Path("raw_data")
# BASE_PATH = Path("raw_data") / "test" # TEST PATH
OUTPUT_PATH = Path("aggregated_data")
# LABELS = ["One", "Two", "Three", "Four", "Five"]
LABEL_SCORE = {"One": 1, "one_two":1, "Two": 2, "two_2": 2, "Three": 3, "three_2":3, "Four": 4, "four_2":4, "Five": 5, "five_2": 5}
# LABEL_SCORE = {"one_test": 1, "two_test": 2, "three_test": 3, "four_test": 4, "five_test": 5} # TEST LABELS

LABELS = list(LABEL_SCORE.keys())
SENSORS = {
    "Accelerometer.csv": "acc_",
    "Linear Accelerometer.csv": "linacc_",
    "Gyroscope.csv": "gyr_",
    "Magnetometer.csv": "mag_"
}
GRANULARITY = '250ms'
START_DATETIME = pd.to_datetime("2025-01-01 00:00:00")

all_data = []
time_offset = pd.to_timedelta(0)

# Function to load and aggregate sensor data
def load_and_aggregate_sensor(file_path, prefix, offset):
    df = pd.read_csv(file_path)

    if "Time (s)" not in df.columns:
        raise ValueError(f"'Time (s)' column not found in {file_path}")

    # Add offset so time is continuous across labels
    df['timestamp'] = START_DATETIME + offset + pd.to_timedelta(df["Time (s)"], unit='s')
    df = df.set_index('timestamp')
    df = df.drop(columns=["Time (s)"])
    df = df.rename(columns=lambda col: prefix + col.strip().split(" ")[0].lower())
    return df.resample(GRANULARITY).mean()

# Loop through each label and process sensors
for label, score in LABEL_SCORE.items():
    print(f"Processing label: {label}")
    folder_path = BASE_PATH / label
    dfs = []

    # Determine duration of this label block from a reference sensor
    ref_file = folder_path / "Accelerometer.csv"
    duration = pd.read_csv(ref_file)["Time (s)"].max()

    # Process each sensor with current offset
    for sensor_file, prefix in SENSORS.items():
        file_path = folder_path / sensor_file
        sensor_df = load_and_aggregate_sensor(file_path, prefix, time_offset)
        dfs.append(sensor_df)

    # Merge sensors on timestamp
    merged = pd.concat(dfs, axis=1)
    merged["score"] = score

    # # Add one-hot encoded labels
    # for lbl in LABELS:
    #     merged[f"label{lbl}"] = 1 if lbl == label else 0

    # Reset index and forward-fill fully missing rows
    merged = merged.reset_index().rename(columns={"timestamp": "datetime"})
    sensor_cols = [col for col in merged.columns if col.startswith(('acc_', 'linacc_', 'gyr_', 'mag_'))]
    sensor_nan_rows = merged[sensor_cols].isna().all(axis=1)
    merged.loc[sensor_nan_rows, sensor_cols] = merged[sensor_cols].ffill().loc[sensor_nan_rows]

    # Save to separate file
    label_clean = label.lower()
    merged.to_csv(OUTPUT_PATH / f"{GRANULARITY}_label{label_clean}.csv", index=False)
    print(f"Dataset saved as: {GRANULARITY}_label{label_clean}.csv")

    # Collect for combined file
    all_data.append(merged)

    # Update offset
    granularity_timedelta = pd.to_timedelta(GRANULARITY)
    time_offset += pd.to_timedelta(duration, unit='s') + granularity_timedelta

# Save combined dataset
final_df = pd.concat(all_data)
final_df.to_csv(OUTPUT_PATH / f"test_{GRANULARITY}_combined.csv", index=False)
print(f"Combined dataset saved as: {GRANULARITY}_combined.csv")
