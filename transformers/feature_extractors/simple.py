from utils import DEFAULT_COLUMN_CONFIG
from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np

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
    























