import pandas as pd
from pathlib import Path

# Load combined dataset
GRANULARITY = '250ms'
file_path = Path(f"aggregated_data/{GRANULARITY}_combined.csv")
df = pd.read_csv(file_path)

# Select sensor columns
sensor_cols = [col for col in df.columns if col.startswith(('acc_', 'linacc_', 'gyr_', 'mag_'))]

# Compute statistics
summary_stats = df[sensor_cols].describe().T[["count", "mean", "std", "min", "max"]]
summary_stats["% missing"] = df[sensor_cols].isnull().mean() * 100
summary_stats = summary_stats[["count", "% missing", "mean", "std", "min", "max"]].reset_index()
summary_stats = summary_stats.rename(columns={"index": "Feature"})

# Print results
print(summary_stats)
summary_stats.to_csv(f"aggregated_data/{GRANULARITY}_stats.csv", index=False)