import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
import tsfel

# Default column configuration
DEFAULT_COLUMN_CONFIG = {
    'primary_key': ['ProcessId', 'DateTime'],
    'time_col': 'DateTime',
    'target_col': 'event',
    'protected_cols': ['ProcessId', 'DateTime', 'event']
}

# 1️⃣ Regularity Resampling Transformer
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

# 2️⃣ NA Handling
class NAHandler:
    """Utility class for handling missing values."""
    
    @staticmethod
    def handle_event_column(df, target_col='event', event_fill=0):
        """Specialized handling for the event column."""
        if target_col in df.columns and event_fill is not None:
            df = df.copy()
            df[target_col] = df[target_col].fillna(event_fill).astype(int)
        return df
    
    @staticmethod
    def split_and_process(X, process_func, column_config=None, event_fill=0):
        """Split data into processable and protected columns, apply processing."""
        X = X.copy()
        config = column_config or DEFAULT_COLUMN_CONFIG
        
        # Get protected columns
        protected_cols = config['protected_cols']
        existing_protected = [col for col in protected_cols if col in X.columns]
        
        # Get processable columns
        processable_cols = [col for col in X.columns if col not in existing_protected]
        
        if not processable_cols:
            return X
        
        # Process columns
        processed = process_func(X[processable_cols])
        protected = X[existing_protected]
        
        # Combine results
        result = pd.concat([protected, processed], axis=1)
        return NAHandler.handle_event_column(
            result, 
            target_col=config['target_col'], 
            event_fill=event_fill
        )

class NAInterpolator(BaseEstimator, TransformerMixin):
    """Interpolates missing values."""
    
    def __init__(self, method='linear', event_fill=0, column_config=None):
        self.method = method
        self.event_fill = event_fill
        self.column_config = column_config or DEFAULT_COLUMN_CONFIG

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        kwargs = ({'method': self.method} if isinstance(self.method, str) 
                 else {'method': self.method[0], 'order': self.method[1]})
        
        return NAHandler.split_and_process(
            X,
            process_func=lambda df: df.interpolate(**kwargs),
            column_config=self.column_config,
            event_fill=self.event_fill
        )

class NAForwardFill(BaseEstimator, TransformerMixin):
    """Fills missing values using forward fill strategy."""
    
    def __init__(self, event_fill=0, column_config=None):
        self.event_fill = event_fill
        self.column_config = column_config or DEFAULT_COLUMN_CONFIG

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return NAHandler.split_and_process(
            X,
            process_func=lambda df: df.ffill(),
            column_config=self.column_config,
            event_fill=self.event_fill
        )

class NABackwardFill(BaseEstimator, TransformerMixin):
    """Fills missing values using backward fill strategy."""
    
    def __init__(self, event_fill=0, column_config=None):
        self.event_fill = event_fill
        self.column_config = column_config or DEFAULT_COLUMN_CONFIG

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return NAHandler.split_and_process(
            X,
            process_func=lambda df: df.bfill(),
            column_config=self.column_config,
            event_fill=self.event_fill
        )

class ImputationWrapper(BaseEstimator, TransformerMixin):
    """
    Wrapper for different imputation strategies.
    
    Parameters
    ----------
    strategy : str, default='interpolate'
        The imputation strategy to use:
        - 'interpolate': Uses NAInterpolator
        - 'ffill': Uses NAForwardFill
        - 'bfill': Uses NABackwardFill
    method : str or tuple, default='linear'
        The interpolation method if strategy='interpolate'
    event_fill : int, default=0
        Value to fill NAs in event column
    column_config : dict, default=None
        Configuration for column names
    """
    def __init__(self, params=('interpolate', 'linear'), 
                 event_fill=0, column_config=None):
        self.params = params
        self.strategy = params[0]
        self.method = params[1]
        self.event_fill = event_fill
        self.column_config = column_config or DEFAULT_COLUMN_CONFIG
        self._imputer = None

    def fit(self, X, y=None):
        if self.strategy == 'interpolate':
            self._imputer = NAInterpolator(
                method=self.method,
                event_fill=self.event_fill,
                column_config=self.column_config
            )
        elif self.strategy == 'ffill':
            self._imputer = NAForwardFill(
                event_fill=self.event_fill,
                column_config=self.column_config
            )
        elif self.strategy == 'bfill':
            self._imputer = NABackwardFill(
                event_fill=self.event_fill,
                column_config=self.column_config
            )
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")
        
        self._imputer.fit(X, y)
        return self

    def transform(self, X):
        if self._imputer is None:
            raise ValueError("ImputationWrapper not fitted")
        return self._imputer.transform(X)

# 3️⃣ Feature Extraction
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
# 4️⃣ Feature Selection
class FeatureSelectionWrapper(BaseEstimator, TransformerMixin):
    """
    Wrapper for different feature selection strategies.
    
    Parameters
    ----------
    strategy : str, default='correlation'
        Feature selection strategy ('correlation' or 'pca')
    threshold : float, default=0.9
        Threshold for correlation-based selection
    variance_threshold : float, default=0.95
        Threshold for PCA-based selection
    column_config : dict, default=None
        Configuration for column names
    """
    def __init__(self, strategy='correlation', threshold=0.9, column_config=None):
        self.strategy = strategy
        self.threshold = threshold
        self.column_config = column_config or DEFAULT_COLUMN_CONFIG
        self._selector = None

    def fit(self, X, y=None):
        if self.strategy == 'correlation':
            self._selector = CorrelationFeatureSelector(
                threshold=self.threshold,
                column_config=self.column_config
            )
        elif self.strategy == 'pca':
            self._selector = PCAFeatureSelector(
                variance_threshold=self.threshold,
                column_config=self.column_config
            )
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")
        
        self._selector.fit(X, y)
        return self

    def transform(self, X):
        if self._selector is None:
            raise ValueError("FeatureSelectionWrapper not fitted")
        return self._selector.transform(X)
class CorrelationFeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(self, threshold=0.9, column_config=None):
        self.threshold = threshold
        self.column_config = column_config or DEFAULT_COLUMN_CONFIG
        self.drop_features = []

    def fit(self, X, y=None):
        # Exclude protected columns from correlation analysis
        numeric_cols = [col for col in X.select_dtypes(include=[np.number]).columns 
                       if col not in self.column_config['protected_cols']]
        
        corr_matrix = X[numeric_cols].corr().abs()
        upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

        self.drop_features = [col for col in upper_tri.columns 
                            if any(upper_tri[col] > self.threshold)]
        return self

    def transform(self, X):
        return X.drop(columns=self.drop_features, errors="ignore")

class PCAFeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(self, variance_threshold=0.95, column_config=None):
        self.variance_threshold = variance_threshold
        self.column_config = column_config or DEFAULT_COLUMN_CONFIG
        self.pca = None
        self.num_components = None

    def fit(self, X, y=None):
        # Select numeric columns while excluding protected ones
        feature_cols = [col for col in X.select_dtypes(include=[np.number]).columns 
                       if col not in self.column_config['protected_cols']]
        X_numeric = X[feature_cols]

        # Fit PCA
        self.pca = PCA()
        self.pca.fit(X_numeric)
        explained_variance_ratio = np.cumsum(self.pca.explained_variance_ratio_)

        # Determine number of components
        self.num_components = np.argmax(explained_variance_ratio >= self.variance_threshold) + 1
        return self

    def transform(self, X):
        X = X.copy()
        feature_cols = [col for col in X.select_dtypes(include=[np.number]).columns 
                       if col not in self.column_config['protected_cols']]
        X_numeric = X[feature_cols]

        # Apply PCA transformation
        X_pca = self.pca.transform(X_numeric)[:, :self.num_components]

        # Create DataFrame with PCA components
        pca_cols = [f'PC{i+1}' for i in range(self.num_components)]
        X_pca_df = pd.DataFrame(X_pca, columns=pca_cols, index=X.index)

        # Combine with protected columns
        protected_df = X[self.column_config['protected_cols']]
        X_transformed = pd.concat([protected_df, X_pca_df], axis=1)

        return X_transformed
    

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
            batch_size = min(1000, len(windows))
            for start_idx in range(0, len(windows), batch_size):
                end_idx = min(start_idx + batch_size, len(windows))
                batch_windows = windows[start_idx:end_idx]
                
                try:
                    # Extract features for the entire batch at once
                    features_batch = tsfel.time_series_features_extractor(
                        self._cfg,
                        batch_windows,
                        fs=1.0
                    )

                    # Update feature arrays efficiently
                    for feat_name in self.feature_names:
                        feature_columns[f"{col}_{feat_name}"][
                            start_idx + self.n_lags:end_idx + self.n_lags
                        ] = features_batch[feat_name].values
                        
                except Exception as e:
                    print(f"Error processing batch at index {start_idx}: {e}")
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
                       if col not in protected_cols]
        
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