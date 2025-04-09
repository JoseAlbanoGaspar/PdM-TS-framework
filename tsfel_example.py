import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from transformers import FeatureSelectionWrapper, ImputationWrapper, TSFELLagFeatureExtractor
import time
from FE_configuration import feature_extraction_preprocessing
from utils import train_test_split_by_time





# Load dataset
print("Loading data...")
raw_df = pd.read_pickle("Datasets/final_dataset.pkl")

'''
# Count records per ProcessId
process_counts = raw_df.groupby('ProcessId').size()

print("\nRecords per ProcessId:")
print("-" * 50)
print(process_counts)
print("\nSummary Statistics:")
print("-" * 50)
print(f"Minimum records: {process_counts.min()}")
print(f"Maximum records: {process_counts.max()}")
print(f"Number of unique ProcessIds: {len(process_counts)}")

#print(f"Loaded {raw_df.shape[0]} rows and {raw_df.shape[1]} columns.")
'''

# Create synthetic dataset
'''dates = pd.date_range(start='2024-01-01', periods=10, freq='H')
process_ids = [1, 1, 1, 1, 1, 2, 2, 2, 2, 2]

synthetic_data = {
    'ProcessId': process_ids,
    'DateTime': dates,
    'Pressure': [100, 102, 98, 103, 97, 95, 94, 96, 93, 98],
    'event': [0, 0, 0, 1, 0, 0, 0, 1, 0, 0]
}'''


'''
# Create synthetic dataset
dates = pd.date_range(start='2024-01-01', periods=200, freq='H')
process_ids = [1] * 100 + [2] * 100  # 100 entries for each process ID

# Generate realistic pressure values with some noise and trends
np.random.seed(42)  # For reproducibility
base_pressure_1 = 100
base_pressure_2 = 95
pressure = []

# Process 1: Pressure around 100 with gradual increase and noise
for i in range(100):
    trend = i * 0.02  # Slight upward trend
    noise = np.random.normal(0, 1)
    pressure.append(base_pressure_1 + trend + noise)

# Process 2: Pressure around 95 with periodic behavior and noise
for i in range(100):
    periodic = 2 * np.sin(i * 0.1)  # Add periodic behavior
    noise = np.random.normal(0, 0.8)
    pressure.append(base_pressure_2 + periodic + noise)

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
    'event': events
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

train_df, test_df = train_test_split_by_time(raw_df, time_col='DateTime', id_col='ProcessId', train_ratio=0.2)
X_train, y_train = train_df.drop(columns=['event']), train_df['event']
X_test, y_test = test_df.drop(columns=['event']), test_df['event']

feat_extract_config = feature_extraction_preprocessing(train_df, COLUMN_CONFIG)




'''
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
        n_lags=8,
        domains=['statistical'],  # Using only temporal features for test
        column_config=COLUMN_CONFIG
    )),
    ('feature_selection', FeatureSelectionWrapper(strategy='correlation', column_config=COLUMN_CONFIG))
    ])

# Measure the time taken for the fit_transform operation
start_time = time.time()
transformed_df = pipeline.fit_transform(raw_df)
end_time = time.time()

# Print the elapsed time
elapsed_time = end_time - start_time
print(f"\nTime taken for fit_transform: {elapsed_time:.2f} seconds")


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
'''