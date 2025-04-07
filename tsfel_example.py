import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from transformers import FeatureSelectionWrapper, ImputationWrapper, TSFELLagFeatureExtractor

# Load dataset
print("Loading data...")
raw_df = pd.read_pickle("Datasets/final_dataset.pkl")
print(f"Loaded {raw_df.shape[0]} rows and {raw_df.shape[1]} columns.")

'''
# Create synthetic dataset
dates = pd.date_range(start='2024-01-01', periods=10, freq='H')
process_ids = [1, 1, 1, 1, 1, 2, 2, 2, 2, 2]

synthetic_data = {
    'ProcessId': process_ids,
    'DateTime': dates,
    'Pressure': [100, 102, 98, 103, 97, 95, 94, 96, 93, 98],
    'Temperature': [25, 26, 24, 27, 23, 22, 21, 23, 20, 24],
    'event': [0, 0, 0, 1, 0, 0, 0, 1, 0, 0]
}

raw_df = pd.DataFrame(synthetic_data)

'''
# Define column configuration
COLUMN_CONFIG = {
    'primary_key': ['ProcessId', 'DateTime'],
    'time_col': 'DateTime',
    'target_col': 'event',
    'protected_cols': ['ProcessId', 'DateTime', 'event']
}

# Print initial data info
print("\nInitial Data:")
print("-" * 50)
print(f"Shape: {raw_df.shape}")
print("\nSample of raw data:")
print(raw_df[['ProcessId', 'DateTime', 'event']].head())

# Create and run pipeline
print("\nExtracting features...")
pipeline = Pipeline([
    ('imputation', ImputationWrapper(params=('interpolate', 'linear'), column_config=COLUMN_CONFIG)),
    ('feature_extraction', TSFELLagFeatureExtractor(
        n_lags=2,
        domains=['temporal'],  # Using only temporal features for test
        column_config=COLUMN_CONFIG
    )),
    ('feature_selection', FeatureSelectionWrapper(strategy='pca', column_config=COLUMN_CONFIG))
    ])

# Transform data
transformed_df = pipeline.fit_transform(raw_df)

# Print results
print("\nTransformation Results:")
print("-" * 50)
print(f"Original shape: {raw_df.shape}")
print(f"Transformed shape: {transformed_df.shape}")

# Show sample of results
print("\nSample of transformed data:")
print("-" * 50)
# Show first 3 rows with original columns and first 5 generated features
feature_cols = [col for col in transformed_df.columns 
                if col not in COLUMN_CONFIG['protected_cols']]
print("\nFirst 3 rows with 5 features:")
print(transformed_df.head(3))
