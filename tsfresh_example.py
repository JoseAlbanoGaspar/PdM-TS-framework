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
#raw_df = pd.read_pickle("Datasets/final_dataset.pkl")

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
timeseries = raw_df.drop(columns=['event'])

# Define column configuration
COLUMN_CONFIG = {
    'primary_key': ['ProcessId', 'DateTime'],
    'time_col': 'DateTime',
    'target_col': 'event',
    'protected_cols': ['ProcessId', 'DateTime', 'event']
}

# Create and run pipeline
print("\nExtracting features...")
pipeline = Pipeline([
    ('imputation', ImputationWrapper(params=('interpolate', 'linear'), column_config=COLUMN_CONFIG)),
    ('feature_extraction', TSFreshLagFeatureExtractor(
        n_lags=4,
        default_fc_parameters=EfficientFCParameters(),
        column_config=COLUMN_CONFIG,
        features=None,  # Use all features by default
        #features=top_features,  # Use top features if available
    ))
    #('feature_selection', FeatureSelectionWrapper(strategy='correlation', column_config=COLUMN_CONFIG))
    ])

# Measure the time taken for the fit_transform operation
start_time = time.time()
transformed_df = pipeline.fit_transform(raw_df)
end_time = time.time()

print("\nExtracted features:")
print("-" * 50)     
print(transformed_df.head())
print(transformed_df.shape)

