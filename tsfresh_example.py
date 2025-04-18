import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from transformers import FeatureSelectionWrapper, ImputationWrapper
import time
from FE_configuration import feature_extraction_preprocessing
from transformers.feature_extraction import TSFreshLagFeatureExtractor
from utils import train_test_split_by_time
from tsfresh import extract_features, select_features
from tsfresh.utilities.dataframe_functions import roll_time_series
from tsfresh.feature_extraction import  EfficientFCParameters

# Load dataset
print("Loading data...")
raw_df = pd.read_pickle("Datasets/final_dataset.pkl")
'''
dates = pd.date_range(start='2024-01-01', periods=10, freq='H')
process_ids = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1] 

synthetic_data = {
    'ProcessId': process_ids,
    'DateTime': dates,
    'Pressure': [100, 102, 98, 103, 97, 95, 94, 96, 93, 98],
    'Temp': [100, 102, 98, 103, 97, 95, 94, 96, 93, 98],

    'event': [0, 0, 0, 1, 0, 0, 0, 1, 0, 0]
}
raw_df = pd.DataFrame(synthetic_data)
'''
'''
# Create synthetic dataset
dates = pd.date_range(start='2024-01-01', periods=200, freq='H')
process_ids = [1] * 100 + [2] * 100  # 100 entries for each process ID

# Generate realistic pressure values with some noise and trends
np.random.seed(42)  # For reproducibility
base_pressure_1 = 100
base_pressure_2 = 95
pressure = []
temp = []
# Process 1: Pressure around 100 with gradual increase and noise
for i in range(200):
    trend = i * 0.02  # Slight upward trend
    noise = np.random.normal(0, 1)
    pressure.append(base_pressure_1 + trend + noise)

# Process 2: Pressure around 95 with periodic behavior and noise
for i in range(200):
    periodic = 2 * np.sin(i * 0.1)  # Add periodic behavior
    noise = np.random.normal(0, 0.8)
    temp.append(base_pressure_2 + periodic + noise)

# Generate events with some patterns
events = []
# Process 1 events: every 20 hours
events.extend([1 if i % 20 == 19 else 0 for i in range(100)])
# Process 2 events: every 25 hours
events.extend([1 if i % 25 == 24 else 0 for i in range(100)])

synthetic_data = {
    'ProcessId': process_ids,
    'DateTime': dates,
    'Pressure': pressure,
    'Temp': temp,
    'event': events
}

raw_df = pd.DataFrame(synthetic_data)
'''
# Define column configuration
COLUMN_CONFIG = {
    'primary_key': ['ProcessId', 'DateTime'],
    'time_col': 'DateTime',
    'target_col': 'event',
    'protected_cols': ['ProcessId', 'DateTime', 'event'],
    'id_col': 'ProcessId'
}

# Configuration for the feature extraction phase
N_FEATURES = 10
N_TSFEL_FEATURES = 10
N_TSFRESH_FEATURES = 10

train_df, test_df = train_test_split_by_time(raw_df, time_col='DateTime', id_col='ProcessId', train_ratio=0.1)
X_train, y_train = train_df.drop(columns=['event']), train_df['event']
X_test, y_test = test_df.drop(columns=['event']), test_df['event']


tsfel_config_file, top_features, tsfresh_fc_parameters = feature_extraction_preprocessing(train_df, COLUMN_CONFIG, N_FEATURES, N_TSFEL_FEATURES, N_TSFRESH_FEATURES)

#tsfresh_features = ['sum_of_reoccurring_data_points', 'cwt_coefficients__coeff_3__w_2__widths_(2, 5, 10, 20)', 'energy_ratio_by_chunks__num_segments_10__segment_focus_1', 'sum_of_reoccurring_values', 'change_quantiles__f_agg_"var"__isabs_False__qh_0.8__ql_0.0', 'sum_values', 'augmented_dickey_fuller__attr_"teststat"__autolag_"AIC"', 'benford_correlation', 'energy_ratio_by_chunks__num_segments_10__segment_focus_2', 'change_quantiles__f_agg_"mean"__isabs_False__qh_0.8__ql_0.2']

#top_features = ['EventCode', 'TPressProdReportHardSrel', 'TPressParameterCode_24.0', 'TPressParameterCode_15.0', 'TPressProdReportThickSrel', 'TPressParameterCode_29.0', 'TPressParameterCode_45.0', 'TPressHardSample1', 'TPressParameterCode_27.0', 'TPressParameterCode_13.0']
#tsfresh_fc_parameters = {'sum_of_reoccurring_data_points': None, 'cwt_coefficients': [{'coeff': 3, 'w': 2, 'widths': (2.0, 5.0, 10.0, 20.0)}], 'energy_ratio_by_chunks': [{'num_segments': 10, 'segment_focus': 1}, {'num_segments': 10, 'segment_focus': 2}], 'sum_of_reoccurring_values': None, 'change_quantiles': [{'f_agg': 'var', 'isabs': False, 'qh': 0.8, 'ql': 0}, {'f_agg': 'mean', 'isabs': False, 'qh': 0.8, 'ql': 0.2}], 'sum_values': None, 'augmented_dickey_fuller': [{'attr': 'teststat', 'autolag': 'AIC'}], 'benford_correlation': None}

print(top_features)
print(tsfresh_fc_parameters)

# Create and run pipeline
print("\nExtracting features...")
pipeline = Pipeline([
    ('imputation', ImputationWrapper(params=('interpolate', 'linear'), column_config=COLUMN_CONFIG)),
    ('feature_extraction', TSFreshLagFeatureExtractor(
        n_lags=4,
        default_fc_parameters=tsfresh_fc_parameters,
        column_config=COLUMN_CONFIG,
        features=top_features
    ))
    #('feature_selection', FeatureSelectionWrapper(strategy='correlation', column_config=COLUMN_CONFIG))
    ])

# Measure the time taken for the fit_transform operation
start_time = time.time()
transformed_df = pipeline.fit_transform(train_df)
end_time = time.time()

print("\nExtracted features:")
print("-" * 50)     
print(transformed_df.head())
print(transformed_df.shape)

for col in transformed_df.columns:
    print(col)
print("-" * 50)