import pandas as pd
import numpy as np
from scipy.signal import butter, filtfilt
import argparse

def calculate_smv(df, acc_cols):
    """Calculate Signal Magnitude Vector for acceleration data"""
    # Assuming acc_cols are in order [x, y, z]
    x = df[acc_cols[0]]
    y = df[acc_cols[1]]
    z = df[acc_cols[2]]
    return np.sqrt(x**2 + y**2 + z**2)

def calculate_zcr(signal):
    """Calculate Zero Crossing Rate"""
    return np.sum(np.diff(np.signbit(signal))) / (len(signal) - 1)

def add_features(df, window_size=30):
    """
    Add features to the dataset using a rolling window approach
    
    Parameters:
    -----------
    df : pandas DataFrame
        Input dataset
    window_size : int
        Window size in seconds
    """
    # Print all column names for debugging
    print("Available columns in the dataset:")
    print(df.columns.tolist())
    
    # Calculate sampling rate from the first two timestamps
    sampling_rate = 1 / (df.index[1] - df.index[0]).total_seconds()
    window_samples = int(window_size * sampling_rate)
    
    # Define the specific columns we want to process
    target_columns = [
        'acc_x', 'acc_y', 'acc_z',
        'linacc_x', 'linacc_y', 'linacc_z', 
        'gyr_x', 'gyr_y', 'gyr_z',
        'mag_x', 'mag_y', 'mag_z'
    ]
    
    # Check which target columns exist in the dataset
    available_target_cols = [col for col in target_columns if col in df.columns]
    missing_cols = [col for col in target_columns if col not in df.columns]
    
    print(f"\nTarget columns found: {available_target_cols}")
    if missing_cols:
        print(f"Target columns missing: {missing_cols}")
    
    if not available_target_cols:
        raise ValueError("None of the target sensor columns found in the dataset.")
    
    # Group columns by sensor type (acc, linacc, gyr, mag)
    sensor_groups = {
        'acc': [col for col in available_target_cols if col.startswith('acc_')],
        'linacc': [col for col in available_target_cols if col.startswith('linacc_')],
        'gyr': [col for col in available_target_cols if col.startswith('gyr_')],
        'mag': [col for col in available_target_cols if col.startswith('mag_')]
    }
    
    # Remove empty groups
    sensor_groups = {k: v for k, v in sensor_groups.items() if v}
    
    print("\nGrouped sensor columns:")
    for sensor, cols in sensor_groups.items():
        print(f"{sensor}: {cols}")
    
    # Calculate features for each sensor group
    for sensor_type, cols in sensor_groups.items():
        if len(cols) == 3:  # Only calculate SMV if we have x, y, z components
            # Calculate SMV (Signal Magnitude Vector)
            df[f'{sensor_type}_smv'] = calculate_smv(df, cols)
            
            # Calculate rolling statistics for SMV
            df[f'{sensor_type}_smv_mean'] = df[f'{sensor_type}_smv'].rolling(window_samples).mean()
            df[f'{sensor_type}_smv_std'] = df[f'{sensor_type}_smv'].rolling(window_samples).std()
            df[f'{sensor_type}_smv_var'] = df[f'{sensor_type}_smv'].rolling(window_samples).var()
            
            print(f"Added SMV features for {sensor_type}")
        else:
            print(f"Warning: {sensor_type} doesn't have all 3 axes (x,y,z). Skipping SMV calculation.")
        
        # Calculate ZCR and basic statistics for each axis
        for col in cols:
            # Zero Crossing Rate
            df[f'{col}_zcr'] = df[col].rolling(window_samples).apply(calculate_zcr)
            
            # Basic statistics
            df[f'{col}_mean'] = df[col].rolling(window_samples).mean()
            df[f'{col}_std'] = df[col].rolling(window_samples).std()
            df[f'{col}_var'] = df[col].rolling(window_samples).var()
            
        print(f"Added ZCR and basic statistics for {sensor_type} axes")
    
    return df

def main():
    parser = argparse.ArgumentParser(description='Add features to sensor dataset')
    parser.add_argument('input_file', help='Input CSV file path')
    parser.add_argument('output_file', help='Output CSV file path')
    parser.add_argument('--window_size', type=int, default=30,
                      help='Window size in seconds (default: 30)')
    
    args = parser.parse_args()
    
    # Read the dataset
    print(f"Reading dataset from {args.input_file}")
    df = pd.read_csv(args.input_file, index_col=0)
    df.index = pd.to_datetime(df.index)
    
    # Add features
    df = add_features(df, args.window_size)
    
    # Save the result
    df.to_csv(args.output_file)
    print(f"Features added and saved to {args.output_file}")

if __name__ == "__main__":
    main() 