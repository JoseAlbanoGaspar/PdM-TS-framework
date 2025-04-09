from utils import DEFAULT_COLUMN_CONFIG
from sklearn.base import BaseEstimator, TransformerMixin

import pandas as pd
import numpy as np

class RegularityResampler(BaseEstimator, TransformerMixin):
    """
    Resamples time series data to regular intervals while preserving event signals.
    
    Parameters
    ----------
    freq : str, default='1H'
        Resampling frequency (e.g., '15T' for 15 minutes, '1H' for 1 hour)
    column_config : dict, default=None
        Configuration for column names
    """
    def __init__(self, freq='1H', column_config=None):
        self.freq = freq
        self.column_config = column_config or DEFAULT_COLUMN_CONFIG

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        primary_key = self.column_config['primary_key']
        time_col = self.column_config['time_col']
        target_col = self.column_config['target_col']
        id_col = [col for col in primary_key if col != time_col][0]

        # Ensure unique timestamps
        X = X.drop_duplicates(subset=primary_key, keep='last')
        X = X.sort_values(by=primary_key)

        resampled_dfs = []
        for process_id, group in X.groupby(id_col):
            # Store failure events
            failure_events = group[group[target_col] == 1].copy()

            # Create full time index
            full_index = pd.date_range(
                start=group[time_col].min(),
                end=group[time_col].max(),
                freq=self.freq
            )
            full_index = pd.MultiIndex.from_product(
                [[process_id], full_index],
                names=[id_col, time_col]
            )

            # Resample
            group = group.set_index(primary_key).reindex(full_index)
            group[target_col] = group[target_col].fillna(0)

            # Restore failure events
            for _, failure_row in failure_events.iterrows():
                failure_time = failure_row.name
                if failure_time in group.index:
                    group.at[failure_time, target_col] = 1

            # Forward-fill failures
            group[target_col] = (group[target_col]
                               .replace(0, np.nan)
                               .ffill()
                               .fillna(0)
                               .astype(int))

            resampled_dfs.append(group)

        return pd.concat(resampled_dfs).reset_index()
