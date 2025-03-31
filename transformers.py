import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA


# 1️⃣ Regularity Resampling Transformer
class RegularityResampler(BaseEstimator, TransformerMixin):
    def __init__(self, freq='1H'):
        self.freq = freq  # Resampling frequency

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        # Ensure unique timestamps per ProcessId
        X = X.drop_duplicates(subset=['ProcessId', 'DateTime'], keep='last')

        # Sort by ProcessId and DateTime
        X = X.sort_values(by=['ProcessId', 'DateTime'])

        resampled_dfs = []
        for process_id, group in X.groupby("ProcessId"):
            # Store failure events separately before resampling
            failure_events = group[group['event'] == 1].copy()

            # Create a full time index for this process
            full_index = pd.date_range(start=group["DateTime"].min(),
                                       end=group["DateTime"].max(),
                                       freq=self.freq)
            full_index = pd.MultiIndex.from_product([[process_id], full_index],
                                                    names=["ProcessId", "DateTime"])

            # Resample: Reindexing will introduce NaNs for missing timestamps
            group = group.set_index(["ProcessId", "DateTime"]).reindex(full_index)

            # Restore failure events explicitly (ensuring event = 1 is not lost)
            group['event'] = group['event'].fillna(0)  # Default fill with 0
            for _, failure_row in failure_events.iterrows():
                failure_time = failure_row.name  # Get original timestamp
                if failure_time in group.index:
                    group.at[failure_time, 'event'] = 1  # Restore the failure event

            # Forward-fill failures to avoid losing the signal
            group['event'] = group['event'].replace(0, np.nan).ffill().fillna(0).astype(int)

            resampled_dfs.append(group)

        return pd.concat(resampled_dfs).reset_index()

# 2️⃣ Imputation Transformers

class ImputationWrapper(BaseEstimator, TransformerMixin):
    """
    Wrapper for different imputation strategies.
    
    Parameters:
    -----------
    strategy : str, default='interpolate'
        The imputation strategy to use. Available options:
        - 'interpolate': Uses NAInterpolator with linear interpolation
        - 'ffill': Uses NAForwardFill for forward filling
        - 'bfill': Uses NABackwardFill for backward filling
    method : str or tuple, default='linear'
        The interpolation method to use if strategy='interpolate'.
        Can be a string or a tuple (method_name, order).
    event_fill : int, default=0
        Value to fill NAs in the event column
    """
    def __init__(self, strategy='interpolate', method='linear', event_fill=0):
        self.strategy = strategy
        self.method = method
        self.event_fill = event_fill
        self._imputer = None
        
    def fit(self, X, y=None):
        # Initialize the appropriate imputer based on the strategy
        if self.strategy == 'interpolate':
            self._imputer = NAInterpolator(method=self.method, event_fill=self.event_fill)
        elif self.strategy == 'ffill':
            self._imputer = NAForwardFill(event_fill=self.event_fill)
        elif self.strategy == 'bfill':
            self._imputer = NABackwardFill(event_fill=self.event_fill)
        else:
            raise ValueError(f"Unknown imputation strategy: {self.strategy}. "
                             f"Choose from 'interpolate', 'ffill', or 'bfill'.")
        
        # Fit the imputer
        self._imputer.fit(X, y)
        return self
    
    def transform(self, X):
        # Validate that fit has been called
        if self._imputer is None:
            raise ValueError("ImputationWrapper has not been fitted yet.")
        
        # Apply the transform
        return self._imputer.transform(X)
class NAHandler:
    @staticmethod
    def handle_event_column(df, event_fill=0):
        """
        Specialized handling for the event column.
        
        Args:
            df: DataFrame containing the event column
            event_fill: Value to fill NAs in event column
        
        Returns:
            DataFrame with properly filled event column
        """
        if 'event' in df.columns and event_fill is not None:
            df = df.copy()
            df['event'] = df['event'].fillna(event_fill).astype(int)
        return df
    
    @staticmethod
    def split_and_process(X, process_func, event_fill=0, exclude_cols=None):
        """
        Utility method to:
        1. Exclude specified columns from processing
        2. Apply the specified process_func to remaining columns
        3. Combine the results
        
        Args:
            X: DataFrame to process
            process_func: Function to apply to processable columns
            event_fill: Value to fill NAs in event column
            exclude_cols: List of columns to exclude from processing (default: ['ProcessId', 'DateTime', 'event'])
        """
        X = X.copy()
        
        # Set default exclude columns if none provided
        if exclude_cols is None:
            exclude_cols = ['ProcessId', 'DateTime', 'event']
        
        # Identify columns to exclude that actually exist in X
        exclude_cols = [col for col in exclude_cols if col in X.columns]
        
        # Identify columns to process
        processable_cols = [col for col in X.columns if col not in exclude_cols]
        
        if not processable_cols:
            return X  # Nothing to process
        
        # Process non-excluded columns
        processed_df = process_func(X[processable_cols])
        
        # Create DataFrames for excluded and processed data
        excluded_df = X[exclude_cols] if exclude_cols else pd.DataFrame(index=X.index)
        
        # Concatenate the processed and excluded columns efficiently
        result = pd.concat([excluded_df, processed_df], axis=1)
        
        # Handle event column with dedicated function
        result = NAHandler.handle_event_column(result, event_fill)
            
        return result
class NAInterpolator(BaseEstimator, TransformerMixin):
    def __init__(self, method='linear', event_fill=0):
        self.method = method
        self.event_fill = event_fill

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Handle different method specifications
        if isinstance(self.method, tuple):
            method_name, order = self.method
            interpolate_kwargs = {'method': method_name, 'order': order}
        else:
            interpolate_kwargs = {'method': self.method}

        return NAHandler.split_and_process(
            X,
            process_func=lambda df: df.interpolate(**interpolate_kwargs),
            event_fill=self.event_fill
        )
    
class NAForwardFill(BaseEstimator, TransformerMixin):
    def __init__(self, event_fill=0):
        self.event_fill = event_fill
        
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return NAHandler.split_and_process(
            X,
            process_func=lambda df: df.ffill(),
            event_fill=self.event_fill
        )

class NABackwardFill(BaseEstimator, TransformerMixin):
    def __init__(self, event_fill=0):
        self.event_fill = event_fill
        
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return NAHandler.split_and_process(
            X,
            process_func=lambda df: df.bfill(),
            event_fill=self.event_fill
        )
    

# 3️⃣ Feature Extraction - Lag Features
class LagFeatureExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, n_lags=2, target_col='event'):
        self.n_lags = n_lags  # Number of past values to consider
        self.target_col = target_col  # Column to predict, which should be excluded

    def fit(self, X, y=None):
        return self  # No fitting required

    def transform(self, X):
        X = X.copy()  # Avoid modifying the original data
        X.sort_values(by=['ProcessId', 'DateTime'], inplace=True)

        # Select numeric columns only, EXCLUDING ProcessId, DateTime, and the target variable
        exclude_cols = {'ProcessId', 'DateTime', self.target_col}
        numeric_cols = [col for col in X.select_dtypes(include=[np.number]).columns if col not in exclude_cols]

        # Create a DataFrame to store lag features
        lagged_data = []

        for process_id, group in X.groupby('ProcessId'):
            group = group.sort_values(by='DateTime').copy()

            # Compute rolling mean for numeric columns (excluding the target variable)
            lagged_features = group[numeric_cols].shift(1).rolling(window=self.n_lags, min_periods=1).mean()

            # Prefix column names for clarity
            lagged_features.columns = [f"{col}_lag_mean_{self.n_lags}" for col in lagged_features.columns]

            # Concatenate back with the original group
            group = pd.concat([group, lagged_features], axis=1)

            # Drop first n_lags rows *for this process_id only*
            group = group.iloc[self.n_lags:]

            lagged_data.append(group)

        # Concatenate processed groups
        X_transformed = pd.concat(lagged_data).reset_index(drop=True)

        return X_transformed

# 4️⃣ Feature Selection
class FeatureSelectionWrapper(BaseEstimator, TransformerMixin):
    """
    Wrapper for different feature selection strategies.
    
    Parameters:
    -----------
    strategy : str, default='correlation'
        The feature selection strategy to use. Available options:
        - 'correlation': Uses CorrelationFeatureSelector
        - 'pca': Uses PCAFeatureSelector
    threshold : float, default=0.9
        Threshold value for correlation-based feature selection
    variance_threshold : float, default=0.95
        Variance threshold for PCA-based feature selection
    exclude_cols : list, default=None
        Columns to exclude from feature selection
    """
    def __init__(self, strategy='correlation', threshold=0.9, 
                 variance_threshold=0.95, exclude_cols=None):
        self.strategy = strategy
        self.threshold = threshold
        self.variance_threshold = variance_threshold
        self.exclude_cols = exclude_cols if exclude_cols is not None else ['ProcessId', 'DateTime', 'event']
        self._selector = None
        
    def fit(self, X, y=None):
        # Initialize the appropriate selector based on the strategy
        if self.strategy == 'correlation':
            self._selector = CorrelationFeatureSelector(threshold=self.threshold)
        elif self.strategy == 'pca':
            self._selector = PCAFeatureSelector(
                variance_threshold=self.variance_threshold,
                exclude_cols=self.exclude_cols
            )
        else:
            raise ValueError(f"Unknown feature selection strategy: {self.strategy}. "
                             f"Choose from 'correlation' or 'pca'.")
        
        # Fit the selector
        self._selector.fit(X, y)
        return self
    
    def transform(self, X):
        # Validate that fit has been called
        if self._selector is None:
            raise ValueError("FeatureSelectionWrapper has not been fitted yet.")
        
        # Apply the transform
        return self._selector.transform(X)

# Correlation
class CorrelationFeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(self, threshold=0.9):
        self.threshold = threshold
        self.drop_features = []
        self.protected_columns = ["DateTime", "ProcessId", "event"]

    def fit(self, X, y=None):
        #print("Before correlation analysis:", X.columns)  # Debugging Line

        # Exclude protected columns from correlation analysis
        numeric_cols = [col for col in X.select_dtypes(include=[np.number]).columns if col not in self.protected_columns]
        corr_matrix = X[numeric_cols].corr().abs()
        upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

        self.drop_features = [col for col in upper_tri.columns if any(upper_tri[col] > self.threshold)]
        #print("Columns to drop due to correlation:", self.drop_features)  # Debugging Line

        return self

    def transform(self, X):
        #print("Before dropping correlated features:", X.columns)  # Debugging Line
        X_transformed = X.drop(columns=self.drop_features, errors="ignore")
        #print("After dropping correlated features:", X_transformed.columns)  # Debugging Line
        return X_transformed

# PCA
class PCAFeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(self, variance_threshold=0.95, exclude_cols=None):
        self.variance_threshold = variance_threshold
        self.exclude_cols = exclude_cols if exclude_cols is not None else ['ProcessId', 'DateTime', 'event']
        self.pca = None  # PCA model will be fitted during fit
        self.num_components = None  # Number of components to retain

    def fit(self, X, y=None):
        # Select numeric columns while excluding non-relevant ones
        feature_cols = [col for col in X.select_dtypes(include=[np.number]).columns if col not in self.exclude_cols]
        X_numeric = X[feature_cols]

        # Fit PCA to capture variance
        self.pca = PCA()
        self.pca.fit(X_numeric)
        explained_variance_ratio = np.cumsum(self.pca.explained_variance_ratio_)

        # Determine number of components to retain
        self.num_components = np.argmax(explained_variance_ratio >= self.variance_threshold) + 1
        # print(f"Selected {self.num_components} PCA components to explain {self.variance_threshold * 100}% variance")

        return self

    def transform(self, X):
        X = X.copy()
        feature_cols = [col for col in X.select_dtypes(include=[np.number]).columns if col not in self.exclude_cols]
        X_numeric = X[feature_cols]

        # Apply PCA transformation
        X_pca = self.pca.transform(X_numeric)[:, :self.num_components]

        # Convert to DataFrame with meaningful column names
        pca_cols = [f'PC{i+1}' for i in range(self.num_components)]
        X_pca_df = pd.DataFrame(X_pca, columns=pca_cols, index=X.index)

        # Concatenate back with the excluded columns
        X_transformed = pd.concat([X[self.exclude_cols], X_pca_df], axis=1)

        return X_transformed
