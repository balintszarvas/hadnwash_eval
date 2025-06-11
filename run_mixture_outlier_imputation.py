#!/usr/bin/env python3
"""
Script to apply mixture-based outlier removal on the variables in 250ms_combined.csv,
then impute the removed datapoints using linear interpolation, and visualize
removed outliers and distributions.
"""
import sys
import os
import pandas as pd
import numpy as np
import time

# Add the ML4QS code directory to the path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CODE_DIR = os.path.join(SCRIPT_DIR, 'ML4QS', 'Python3Code')
sys.path.insert(0, CODE_DIR)

from ML4QS.Python3Code.Chapter3.OutlierDetection import DistributionBasedOutlierDetection
from ML4QS.Python3Code.Chapter3.ImputationMissingValues import ImputationMissingValues
from ML4QS.Python3Code.util.VisualizeDataset import VisualizeDataset

# Monkey-patch VisualizeDataset.plot_binary_outliers for safe boolean masking
import matplotlib.pyplot as _plt
import matplotlib.dates as _md

# Mapping sensor prefixes to full sensor names
sensor_map = {
    'acc': 'Accelerometer',
    'linacc': 'Linear Accelerometer',
    'gyr': 'Gyroscope',
    'mag': 'Magnetometer'
}

def _safe_plot_binary_outliers(self, data_table, col, outlier_col):
    # Drop rows with NaNs and ensure boolean dtype
    dt = data_table.dropna(axis=0, subset=[col, outlier_col]).copy()
    dt[outlier_col] = dt[outlier_col].astype(bool)
    fig, ax = _plt.subplots()
    xfmt = _md.DateFormatter('%H:%M')
    ax.xaxis.set_major_formatter(xfmt)
    ax.set_xlabel('time')
    ax.set_ylabel('value')
    # Add title based on sensor prefix and variable
    if '_' in col:
        prefix, var = col.split('_', 1)
        sensor_name = sensor_map.get(prefix, prefix)
        ax.set_title(f"{sensor_name} – {var}")
    else:
        ax.set_title(col)
    # Plot outliers in red and non-outliers in blue
    ax.plot(dt.index[dt[outlier_col]], dt[col][dt[outlier_col]], 'r+', label='outlier')
    ax.plot(dt.index[~dt[outlier_col]], dt[col][~dt[outlier_col]], 'b+', label='no_outlier')
    ax.legend()
    # Custom save with descriptive filename: method, parameter, and variable
    method_name = "mixture_model"
    # Convert quantile to percentage integer (e.g., 1 for 0.01)
    param_quant = THRESHOLD_QUANTILE * 100
    filename = f"{method_name}_q{param_quant}_{col}"
    for ext in ('png', 'pdf'):
        save_path = self.figures_dir / f"{filename}.{ext}"
        _plt.savefig(save_path)
        print(f"Figure saved to {save_path}")
    # Increment the internal figure counter
    self.plot_number += 1
    #_plt.show()

# Override the method on the class
VisualizeDataset.plot_binary_outliers = _safe_plot_binary_outliers

# Threshold quantile to flag outliers (e.g., lowest 1% probabilities)
THRESHOLD_QUANTILE = 0.005

def run_outlier_imputation(csv_path, threshold_quantile=THRESHOLD_QUANTILE):
    # Read the data, parse the first column as datetime index
    df = pd.read_csv(csv_path, parse_dates=[0], index_col=0)
    # Drop duplicate timestamps, keep only the first occurrence
    df = df[~df.index.duplicated(keep='first')]
    df.sort_index(inplace=True)

    # Instantiate outlier detector, imputer, and visualizer
    detector = DistributionBasedOutlierDetection()
    imputer = ImputationMissingValues()
    viz = VisualizeDataset(module_path=__file__)

    # Copy of original data for comparison
    original_df = df.copy()
    start_time = time.time()
    # Process only sensor columns (exclude label columns)
    for col in [c for c in df.columns if not c.startswith('label')]:
        # Extract original series with datetime index
        col_series = df[col].copy()
        n = len(col_series)
        # Create temporary DataFrame with numeric index for outlier detection
        temp_df = pd.DataFrame({col: col_series.values}, index=pd.RangeIndex(start=0, stop=n))

        # Apply mixture model to get probability scores
        temp_df = detector.mixture_model(temp_df, col)
        mix_prob_col = f"{col}_mixture"

        # Determine threshold and flag outliers
        thresh = temp_df[mix_prob_col].quantile(threshold_quantile)
        outlier_col = f"{col}_outlier"
        temp_df[outlier_col] = temp_df[mix_prob_col] < thresh

        # Visualize binary outliers (removed vs kept) using datetime index
        viz_df = pd.DataFrame({col: col_series, outlier_col: temp_df[outlier_col].values}, index=col_series.index)
        viz.plot_binary_outliers(viz_df, col, outlier_col)

        # For acceleration and gyroscope, compare all imputation methods
        if col.startswith(('acc_','mag_')):
            # Identify outlier positions and remove them in the time series
            positions = np.where(temp_df[outlier_col].values)[0]
            removed_series = col_series.copy()
            removed_series.iloc[positions] = np.nan
            df_removed = pd.DataFrame({col: removed_series}, index=col_series.index)

            # Compute imputed series using mean, median, and interpolation
            mean_df = imputer.impute_mean(df_removed.copy(), col)
            median_df = imputer.impute_median(df_removed.copy(), col)
            interp_df = imputer.impute_interpolate(df_removed.copy(), col)

            # Plot original vs mean vs median vs interpolation
            viz.plot_imputed_values(
                df_removed,
                ['original', 'mean', 'median', 'interpolation'],
                col,
                mean_df[col],
                median_df[col],
                interp_df[col]
            )

            # Map the interpolation result back to the main DataFrame
            df[col] = interp_df[col].values
        else:
            # For other columns, simply impute via interpolation
            temp_df.loc[temp_df[outlier_col], col] = np.nan
            temp_df = imputer.impute_interpolate(temp_df, col)
            df[col] = temp_df[col].values
    print(f"Time taken: {time.time() - start_time} seconds")

    # Visualize distributions before and after cleaning (sensor columns only)
    sensor_cols_orig = [c for c in original_df.columns if not c.startswith('label')]
    viz.plot_dataset_boxplot(original_df, sensor_cols_orig)
    sensor_cols_cleaned = [c for c in df.columns if not c.startswith('label')]
    viz.plot_dataset_boxplot(df, sensor_cols_cleaned)

    # Save cleaned data
    output_path = os.path.join(os.path.dirname(csv_path), '250ms_combined_cleaned.csv')
    df.to_csv(output_path)
    print(f"Cleaned data saved to {output_path}")

if __name__ == "__main__":
    # Path to the 250ms combined CSV
    csv_file = os.path.join(SCRIPT_DIR, 'aggregated_data', '250ms_combined.csv')
    run_outlier_imputation(csv_file) 