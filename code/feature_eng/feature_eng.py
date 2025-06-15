##############################################################
#                                                            #
#    Mark Hoogendoorn and Burkhardt Funk (2017)              #
#    Machine Learning for the Quantified Self                #
#    Springer                                                #
#    Chapter 4                                               #
#                                                            #
##############################################################

import sys
import copy
import pandas as pd
import time
from pathlib import Path
import argparse
import numpy as np

from util.VisualizeDataset import VisualizeDataset
from Chapter4.TemporalAbstraction import NumericalAbstraction
from Chapter4.TemporalAbstraction import CategoricalAbstraction
from Chapter4.FrequencyAbstraction import FourierTransformation
from Chapter4.TextAbstraction import TextAbstraction
from Chapter3.DataTransformation import PrincipalComponentAnalysis

# Read the result from the previous chapter, and make sure the index is of the type datetime.
DATA_PATH = Path('./datasets/')
DATASET_FNAME = '250ms_combined_cleaned.csv'
RESULT_FNAME = 'dataset_with_engineered_features.csv'

def print_flags():
    """
    Prints all entries in FLAGS variable.
    """
    for key, value in vars(FLAGS).items():
        print(key + ' : ' + str(value))

# Calculate roll and pitch using accelerometer data
def calculate_roll_pitch(dataset):
    dataset['roll'] = np.arctan2(dataset['acc_y'], dataset['acc_z'])
    dataset['pitch'] = np.arctan2(-dataset['acc_x'], np.sqrt(dataset['acc_y']**2 + dataset['acc_z']**2))
    # Convert radians to degrees for better interpretability
    dataset['roll'] = np.degrees(dataset['roll'])
    dataset['pitch'] = np.degrees(dataset['pitch'])
    return dataset

# Calculate jerk using linear acceleration data
def calculate_jerk(dataset, time_interval):
    dataset['jerk_x'] = dataset['linacc_x'].diff() / time_interval
    dataset['jerk_y'] = dataset['linacc_y'].diff() / time_interval
    dataset['jerk_z'] = dataset['linacc_z'].diff() / time_interval
    dataset['jerk_magnitude'] = np.sqrt(dataset['jerk_x']**2 + dataset['jerk_y']**2 + dataset['jerk_z']**2)
    dataset.fillna(0, inplace=True) # otherwise when calculating the mean, std, etc. with .rolling() we get all NaN values
    return dataset

def apply_pca(dataset):
    PCA = PrincipalComponentAnalysis()
    selected_predictor_cols = [c for c in dataset.columns if (not ('score' in c))]
    n_pcs = 4  # Number of principal components to keep
    dataset = PCA.apply_pca(copy.deepcopy(dataset), selected_predictor_cols, n_pcs)
    return dataset

def main():
    print_flags()
    
    start_time = time.time()
    try:
        dataset = pd.read_csv(DATA_PATH / DATASET_FNAME, index_col=0)
        dataset.index = pd.to_datetime(dataset.index)
    except IOError as e:
        print('File not found, try to run previous crowdsignals scripts first!')
        raise e

    
    time_interval = (dataset.index[1] - dataset.index[0]).total_seconds()  # Calculate time interval in seconds
    dataset = calculate_roll_pitch(dataset)
    dataset = calculate_jerk(dataset, time_interval)
    dataset = apply_pca(dataset)
    DataViz = VisualizeDataset(__file__)
    # DataViz.plot_dataset(dataset, ['roll', 'pitch', 'jerk', 'score'], ['like', 'like', 'like', 'like'], ['line', 'line', 'line', 'points'], 'roll_pitch_jerk_distribution')

    # # Compute the number of milliseconds covered by an instance based on the first two rows
    milliseconds_per_instance = (dataset.index[1] - dataset.index[0]).microseconds/1000
    NumAbs = NumericalAbstraction()
    FreqAbs = FourierTransformation()

    if FLAGS.mode == 'aggregation':
        # Chapter 4: Identifying aggregate attributes.

        # Set the window sizes to the number of instances representing 5 seconds, 30 seconds and 5 minutes
        # window_sizes = [int(float(5000)/milliseconds_per_instance), int(float(0.5*60000)/milliseconds_per_instance), int(float(5*60000)/milliseconds_per_instance)]
        # print(f"Window sizes: {window_sizes}")
        original_features = [
            'acc_x', 'acc_y', 'acc_z',
            'linacc_x', 'linacc_y', 'linacc_z',
            'gyr_x', 'gyr_y', 'gyr_z',
            'mag_x', 'mag_y', 'mag_z',
            'roll', 'pitch',
            'jerk_x', 'jerk_y', 'jerk_z', 'jerk_magnitude'
        ]
        window_sizes = {"roll": int(float(5000)/milliseconds_per_instance),
                        "pitch": int(float(5000)/milliseconds_per_instance),
                        "jerk_x": int(float(2000)/milliseconds_per_instance),
                        "jerk_y": int(float(2000)/milliseconds_per_instance),
                        "jerk_z": int(float(2000)/milliseconds_per_instance),
                        "jerk_magnitude": int(float(2000)/milliseconds_per_instance)}
        
        for feature, window in window_sizes.items():
            # print(f"Feature: {feature}, Window size: {window}")
            print(f"window size for {feature}: {window} instances")
            dataset = NumAbs.abstract_numerical(dataset, [feature], window, 'mean')
            dataset = NumAbs.abstract_numerical(dataset, [feature], window, 'max')
            dataset = NumAbs.abstract_numerical(dataset, [feature], window, 'min')
            dataset = NumAbs.abstract_numerical(dataset, [feature], window, 'range')
            dataset = NumAbs.abstract_numerical(dataset, [feature], window, 'std')

        dataset = NumAbs.abstract_numerical(dataset, ['score'], int(float(5000)/milliseconds_per_instance), 'mode')
        valid_rows_mask = dataset.drop(columns=original_features).notna().any(axis=1)
        dataset = dataset[valid_rows_mask]

        dataset = dataset.iloc[::4]
        dataset = dataset.drop(columns=original_features)
        dataset = dataset.dropna()
        print(len(dataset))
        dataset.to_csv(Path("intermediate_datafiles") / RESULT_FNAME)

  
    if FLAGS.mode == 'final':
        original_features = [
            'acc_x', 'acc_y', 'acc_z',
            'linacc_x', 'linacc_y', 'linacc_z',
            'gyr_x', 'gyr_y', 'gyr_z',
            'mag_x', 'mag_y', 'mag_z',
            'roll', 'pitch',
            'jerk_x', 'jerk_y', 'jerk_z', 'jerk_magnitude', 'score',
            'pca_1', 'pca_2', 'pca_3', 'pca_4'
        ]
        window_sizes = {"roll": int(float(5000)/milliseconds_per_instance),
                        "pitch": int(float(5000)/milliseconds_per_instance),
                        "jerk_x": int(float(2000)/milliseconds_per_instance),
                        "jerk_y": int(float(2000)/milliseconds_per_instance),
                        "jerk_z": int(float(2000)/milliseconds_per_instance),
                        "jerk_magnitude": int(float(2000)/milliseconds_per_instance),
                        "pca_1": int(float(5000)/milliseconds_per_instance),
                        "pca_2": int(float(5000)/milliseconds_per_instance),
                        "pca_3": int(float(5000)/milliseconds_per_instance),
                        "pca_4": int(float(5000)/milliseconds_per_instance)}
        
        for feature, window in window_sizes.items():
            # print(f"Feature: {feature}, Window size: {window}")
            print(f"window size for {feature}: {window} instances")
            if 'pca' in feature:
                dataset = NumAbs.abstract_numerical(dataset, [feature], window, 'mean')
                dataset = NumAbs.abstract_numerical(dataset, [feature], window, 'std')
            else:
                dataset = NumAbs.abstract_numerical(dataset, [feature], window, 'mean')
                dataset = NumAbs.abstract_numerical(dataset, [feature], window, 'max')
                dataset = NumAbs.abstract_numerical(dataset, [feature], window, 'min')
                dataset = NumAbs.abstract_numerical(dataset, [feature], window, 'range')
                dataset = NumAbs.abstract_numerical(dataset, [feature], window, 'std')
   
    #     DataViz.plot_dataset(dataset, ['acc_phone_x', 'gyr_phone_x', 'hr_watch_rate', 'light_phone_lux', 'mag_phone_x', 'press_phone_', 'pca_1', 'label'], ['like', 'like', 'like', 'like', 'like', 'like', 'like','like'], ['line', 'line', 'line', 'line', 'line', 'line', 'line', 'points'])

        print("Calculating frequency domain features...")
        periodic_predictor_cols = ['acc_x','acc_y','acc_z',
                                'linacc_x','linacc_y','linacc_z',
                                'gyr_x','gyr_y', 'gyr_z',
                                'roll', 'pitch']

        fs = float(1000)/milliseconds_per_instance
        dataset = FreqAbs.abstract_frequency(copy.deepcopy(dataset), periodic_predictor_cols, int(float(8000)/milliseconds_per_instance), fs) # 8s window size
        # print(dataset.columns.tolist())

        dataset = NumAbs.abstract_numerical(dataset, ['score'], int(float(2000)/milliseconds_per_instance), 'mode')
        valid_rows_mask = dataset.drop(columns=original_features).notna().any(axis=1)
        dataset = dataset[valid_rows_mask]

        dataset = dataset.drop(columns=original_features)
        print(dataset.shape)
        dataset.to_csv(Path("intermediate_datafiles") / RESULT_FNAME)

    #     DataViz.plot_dataset(dataset, ['acc_phone_x', 'gyr_phone_x', 'hr_watch_rate', 'light_phone_lux', 'mag_phone_x', 'press_phone_', 'pca_1', 'label'], ['like', 'like', 'like', 'like', 'like', 'like', 'like','like'], ['line', 'line', 'line', 'line', 'line', 'line', 'line', 'points'])
    #     print("--- %s seconds ---" % (time.time() - start_time))
  
if __name__ == '__main__':
    # Command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='final',
                        help= "Select what version to run: final, aggregation or freq \
                        'aggregation' studies the effect of several aggeregation methods \
                        'frequency' applies a Fast Fourier transformation to a single variable \
                        'final' is used for the next chapter ", choices=['none', 'aggregation', 'frequency', 'final']) 

    FLAGS, unparsed = parser.parse_known_args()
    
    main()