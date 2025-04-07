import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
import tsfresh

class TSFreshLagFeatureExtractor(BaseEstimator, TransformerMixin):
    """
    Extracts features from lagged windows using tsfresh.
    """
    def __init__(self, n_lags=2, default_fc_parameters='minimal', column_config=None):
        self.n_lags = n_lags
        self.default_fc_parameters = default_fc_parameters
        self.column_config = column_config or {
            'primary_key': ['ProcessId', 'DateTime'],
            'time_col': 'DateTime',
            'target_col': 'event',
            'protected_cols': ['ProcessId', 'DateTime', 'event']
        }

    def fit(self, X, y=None):
        return self

    def _process_group(self, group, numeric_cols):
        """Process a single group of data"""
        if len(group) < self.n_lags:
            return pd.DataFrame()

        # Create time series format required by tsfresh
        ts_data = []
        
        for col in numeric_cols:
            values = group[col].values
            
            # Create sliding windows
            for i in range(len(values) - self.n_lags + 1):
                window = values[i:i + self.n_lags]
                # Add window data points
                for j, val in enumerate(window):
                    ts_data.append({
                        'id': i + self.n_lags - 1,  # Maps to the target row
                        'time': j,  # Position in window
                        'variable': col,  # Original column name
                        'value': val
                    })

        # Convert to DataFrame
        ts_df = pd.DataFrame(ts_data)
        
        if ts_df.empty:
            return pd.DataFrame()

        # Extract features using tsfresh
        features = tsfresh.extract_features(
            ts_df,
            column_id='id',
            column_sort='time',
            column_kind='variable',
            column_value='value',
            default_fc_parameters=tsfresh.feature_extraction.MinimalFCParameters() if self.default_fc_parameters == 'minimal' else None,
            disable_progressbar=True
        )

        # Add original data
        result = group.copy()
        result = pd.concat([
            result.iloc[self.n_lags-1:].reset_index(drop=True),
            features
        ], axis=1)

        return result

    def transform(self, X):
        X = X.copy()
        primary_key = self.column_config['primary_key']
        time_col = self.column_config['time_col']
        protected_cols = self.column_config['protected_cols']
        
        # Get numeric columns
        numeric_cols = [col for col in X.select_dtypes(include=[np.number]).columns 
                       if col not in protected_cols]
        
        # Get id columns
        id_cols = [col for col in primary_key if col != time_col]
        
        # Sort by time
        X = X.sort_values(by=primary_key)
        
        if not id_cols:
            # Single time series
            result = self._process_group(X, numeric_cols)
        else:
            # Multiple time series
            groups = []
            for _, group in X.groupby(id_cols):
                processed_group = self._process_group(group, numeric_cols)
                groups.append(processed_group)
            result = pd.concat(groups, ignore_index=True)
        
        return result.sort_values(by=primary_key).reset_index(drop=True)

# Load dataset and take first 20 rows
print("Loading data...")
raw_df = pd.read_pickle("Datasets/final_dataset.pkl").head(20)

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
    ('feature_extraction', TSFreshLagFeatureExtractor(
        n_lags=2,
        default_fc_parameters='minimal',  # Using minimal feature set for test
        column_config=COLUMN_CONFIG
    ))
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
feature_cols = [col for col in transformed_df.columns 
                if col not in COLUMN_CONFIG['protected_cols']]
print("\nFirst 3 rows with first 5 features:")
print(transformed_df[COLUMN_CONFIG['protected_cols'] + feature_cols[:5]].head(3))