from utils import DEFAULT_COLUMN_CONFIG
from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np

from sklearn.impute import SimpleImputer
import tsfel


class LagFeatureExtractor(BaseEstimator, TransformerMixin):
    """
    Extracts lagged features from time series data.
    
    Parameters
    ----------
    n_lags : int, default=2
        Number of lagged values to consider
    column_config : dict, default=None
        Configuration for column names. Supports both single and multiple time series.
    """
    def __init__(self, n_lags=2, column_config=None):
        self.n_lags = n_lags
        self.column_config = column_config or DEFAULT_COLUMN_CONFIG

    def fit(self, X, y=None):
        return self

    def _compute_lagged_features(self, data, numeric_cols):
        """Helper method to compute lagged features for a group of data"""
        lagged_features = (data[numeric_cols]
                         .shift(1)
                         .rolling(window=self.n_lags, min_periods=1)
                         .mean())
        
        lagged_features.columns = [f"{col}_lag_mean_{self.n_lags}" 
                                 for col in numeric_cols]
        return lagged_features

    def transform(self, X):
        X = X.copy()
        primary_key = self.column_config['primary_key']
        time_col = self.column_config['time_col']
        protected_cols = self.column_config['protected_cols']

        # Get id columns (all primary key columns except time)
        id_cols = [col for col in primary_key if col != time_col]
        
        # Get numeric columns for feature generation
        numeric_cols = [col for col in X.select_dtypes(include=[np.number]).columns 
                       if col not in protected_cols]

        if not id_cols:
            # Case 1: Single time series (only datetime in primary key)
            X = X.sort_values(by=time_col)
            lagged_features = self._compute_lagged_features(X, numeric_cols)
            result = pd.concat([X, lagged_features], axis=1)
            result = result.iloc[self.n_lags:]
        else:
            # Case 2: Multiple time series
            X.sort_values(by=primary_key, inplace=True)
            lagged_data = []
            
            for _, group in X.groupby(id_cols):
                group = group.sort_values(by=time_col).copy()
                lagged_features = self._compute_lagged_features(group, numeric_cols)
                group = pd.concat([group, lagged_features], axis=1)
                group = group.iloc[self.n_lags:]
                lagged_data.append(group)
            
            result = pd.concat(lagged_data, axis=0)

        return result.reset_index(drop=True)
    

class TSFELLagFeatureExtractor(BaseEstimator, TransformerMixin):
    """
    Extracts features from lagged windows using TSFEL.
    """
    def __init__(self, n_lags=2, domains=['temporal'], column_config=None):
        self.n_lags = n_lags
        self.domains = domains if isinstance(domains, list) else [domains]
        self.column_config = column_config or DEFAULT_COLUMN_CONFIG
        self._cfg = None
        self.feature_names = None

    def fit(self, X, y=None):
        self._cfg = tsfel.get_features_by_domain(self.domains)
        # Pre-compute feature names
        sample_window = np.zeros((self.n_lags, 1))
        sample_features = tsfel.time_series_features_extractor(
            self._cfg, sample_window, fs=1.0
        )
        self.feature_names = sample_features.columns
        return self

    def _process_group(self, group, numeric_cols):
        """Process a single group of data efficiently using vectorized operations"""
        result = group.copy()
        n_rows = len(group)
        
        if n_rows < self.n_lags:
            return pd.DataFrame()  # Return empty frame if not enough data
        
        # Create feature columns more efficiently
        feature_columns = {}
        
        for col in numeric_cols:
            values = group[col].values
            # Create sliding windows using stride tricks for better memory efficiency
            windows = np.lib.stride_tricks.sliding_window_view(
                values,
                window_shape=self.n_lags
            )

            # Pre-allocate arrays for features
            for feat_name in self.feature_names:
                feature_columns[f"{col}_{feat_name}"] = np.full(n_rows, np.nan)
            
            # Process windows in batches for better performance
            for i in range(len(windows)):
                # Extract features for each window
                window = windows[i]

                try:
                    features = tsfel.time_series_features_extractor(
                        self._cfg,
                        window.reshape(-1, 1),
                        fs=1.0,
                    )
                    #print(f"Extracted features for window {i}:")
                    #print(features)
                    # Update feature columns with extracted features
                    for feat_name in self.feature_names:
                        feature_columns[f"{col}_{feat_name}"][i + self.n_lags] = features[feat_name].values[0]
                except Exception as e:
                    print(f"Error processing window at index {i}: {e}")
                    print(f"Window shape: {window.shape}, Window values: {window}")
                    continue
        
        # Create features DataFrame all at once
        features_df = pd.DataFrame(
            feature_columns,
            index=result.index
        )
        
        # Combine with original data and remove rows without features
        result = pd.concat([result, features_df], axis=1)
        return result.iloc[self.n_lags:]
    

    def transform(self, X):
        X = X.copy()
        primary_key = self.column_config['primary_key']
        time_col = self.column_config['time_col']
        protected_cols = self.column_config['protected_cols']
        
        # Get numeric columns excluding protected ones
        numeric_cols = [col for col in X.select_dtypes(include=[np.number]).columns 
                       if col not in protected_cols][:10]
        
        print(f"Numeric columns: {numeric_cols}")
        
        # Get id columns
        id_cols = [col for col in primary_key if col != time_col]
        
        # Sort by time within each group
        X = X.sort_values(by=primary_key)
        
        if not id_cols:
            # Single time series case
            result = self._process_group(X, numeric_cols)
        else:
            # Multiple time series case
            groups = []
            for _, group in X.groupby(id_cols):
                processed_group = self._process_group(group, numeric_cols)
                groups.append(processed_group)
            result = pd.concat(groups, ignore_index=True)
        print(f"Result:\n {result}")
        
        # Add this after the print:
        non_protected_cols = [col for col in result.columns if col not in protected_cols]
        # Drop columns that are all NA
        all_na_cols = result[non_protected_cols].columns[result[non_protected_cols].isna().all()].tolist()
        if all_na_cols:
            print(f"Dropping columns with all NA values:\n{all_na_cols}")
            result = result.drop(columns=all_na_cols)
            non_protected_cols = [col for col in result.columns if col not in protected_cols]

        na_cols = result[non_protected_cols].columns[result[non_protected_cols].isna().any()].tolist()
        if na_cols:
            print(f"Columns with NA values:\n{na_cols}")
            print("\nNA count per column:")
            print(result[na_cols].isna().sum())

        # Split data into protected and non-protected columns
        protected_data = result[protected_cols]
        non_protected_data = result.drop(columns=protected_cols)
        
        # Apply SimpleImputer to non-protected columns
        imputer = SimpleImputer(strategy='mean')
        imputed_data = imputer.fit_transform(non_protected_data)
        
        # Convert back to DataFrame with original column names
        imputed_df = pd.DataFrame(
            imputed_data, 
            columns=non_protected_data.columns,
            index=result.index
        )
        
        # Combine protected and imputed data
        final_result = pd.concat([protected_data, imputed_df], axis=1)
        
        return final_result[result.columns]  # Preserve original column order
