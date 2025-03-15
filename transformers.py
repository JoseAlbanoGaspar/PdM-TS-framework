import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA

from sklearn.pipeline import Pipeline
import sys # only for debugging purposes - forcing prints in notebooks
#sys.stdout.flush()  # Force it to display in Jupyter

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

        # Resampling process
        resampled_dfs = []
        for process_id, group in X.groupby("ProcessId"):
            # Create a full time index for this process
            full_index = pd.date_range(start=group["DateTime"].min(),
                                       end=group["DateTime"].max(),
                                       freq=self.freq)
            full_index = pd.MultiIndex.from_product([[process_id], full_index],
                                                    names=["ProcessId", "DateTime"])

            # Set index, reindex to fill missing timestamps
            group = group.set_index(["ProcessId", "DateTime"]).reindex(full_index)

            resampled_dfs.append(group)

        return pd.concat(resampled_dfs).reset_index()

# 2️⃣ Imputation Transformers
class NAHandler:
    @staticmethod
    def split_and_process(X, process_func, event_fill=0):
        """
        Utility method to:
        1. Split a dataframe into event column and other columns
        2. Apply the specified process_func to non-event columns
        3. Fill event column with event_fill value
        4. Combine the results
        
        Args:
            X: DataFrame to process
            process_func: Function to apply to non-event columns
            event_fill: Value to fill NAs in event column
        """
        X = X.copy()
        
        # Check if event column exists
        if 'event' in X.columns:
            # Split event column from the rest
            event_col = X['event'].copy()
            rest_df = X.drop(columns=['event'])
            
            # Process non-event columns
            processed_df = process_func(rest_df)
            
            # Fill event column NAs
            event_col = event_col.fillna(event_fill).astype(int)
            
            # Combine results
            processed_df['event'] = event_col
            return processed_df
        else:
            # If no event column, process everything
            return process_func(X)

class NAInterpolator(BaseEstimator, TransformerMixin):
    def __init__(self, method='linear', event_fill=0):
        self.method = method
        self.event_fill = event_fill

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return NAHandler.split_and_process(
            X,
            process_func=lambda df: df.interpolate(method=self.method),
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

            # Drop first `n_lags` rows *for this process_id only*
            group = group.iloc[self.n_lags:]

            lagged_data.append(group)

        # Concatenate processed groups
        X_transformed = pd.concat(lagged_data).reset_index(drop=True)

        return X_transformed

# 4️⃣ Feature Selection - Correlation
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

# 5️⃣ Feature Selection - PCA
class PCAFeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(self, variance_threshold=0.95, exclude_cols=None):
        self.variance_threshold = variance_threshold
        self.exclude_cols = exclude_cols if exclude_cols is not None else ['ProcessId', 'DateTime', 'event']
        self.pca = None  # PCA model will be fitted during `fit`
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
    

'''
# test the transformers

# Load dataset
df = pd.read_pickle("Datasets/final_dataset.pkl")

# Sort to ensure correct time order
df.sort_values(by=["ProcessId", "DateTime"], inplace=True)

pipeline = Pipeline(steps=[('regularity', RegularityResampler()), ('imputation', NAInterpolator()),('feature_extraction', LagFeatureExtractor(n_lags=3)), ('feature_selection', CorrelationFeatureSelector())])

df_transformed = pipeline.fit_transform(df)
print(df_transformed['event'].unique())
'''