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

    def transform(self, X, y=None):
        X = X.copy()
        y = y.copy() if y is not None else None

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

            # Restore failure events explicitly (ensuring `event = 1` is not lost)
            group['event'] = group['event'].fillna(0)  # Default fill with 0
            for _, failure_row in failure_events.iterrows():
                failure_time = failure_row.name  # Get original timestamp
                if failure_time in group.index:
                    group.at[failure_time, 'event'] = 1  # Restore the failure event

            # Forward-fill failures to avoid losing the signal
            group['event'] = group['event'].replace(0, np.nan).ffill().fillna(0).astype(int)

            resampled_dfs.append(group)

        X_transformed = pd.concat(resampled_dfs).reset_index()

        # Return both X and y
        return X_transformed.drop(columns=['event']), group['event'] if y is None else y


# 2️⃣ Imputation Transformers

class NAHandler:
    @staticmethod
    def split_and_process(X, process_func, event_fill=0):
        X = X.copy()
        
        if 'event' in X.columns:
            event_col = X['event'].copy()
            rest_df = X.drop(columns=['event'])
            processed_df = process_func(rest_df)
            event_col = event_col.fillna(event_fill).astype(int)
            processed_df['event'] = event_col
            return processed_df
        else:
            return process_func(X)


class NAInterpolator(BaseEstimator, TransformerMixin):
    def __init__(self, method='linear', event_fill=0):
        self.method = method
        self.event_fill = event_fill

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X_transformed = NAHandler.split_and_process(
            X,
            process_func=lambda df: df.interpolate(method=self.method),
            event_fill=self.event_fill
        )
        
        return X_transformed.drop(columns=['event']), X_transformed['event'] if y is None else y


class NAForwardFill(BaseEstimator, TransformerMixin):
    def __init__(self, event_fill=0):
        self.event_fill = event_fill
        
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X_transformed = NAHandler.split_and_process(
            X,
            process_func=lambda df: df.ffill(),
            event_fill=self.event_fill
        )
        
        return X_transformed.drop(columns=['event']), X_transformed['event'] if y is None else y


class NABackwardFill(BaseEstimator, TransformerMixin):
    def __init__(self, event_fill=0):
        self.event_fill = event_fill
        
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X_transformed = NAHandler.split_and_process(
            X,
            process_func=lambda df: df.bfill(),
            event_fill=self.event_fill
        )
        
        return X_transformed.drop(columns=['event']), X_transformed['event'] if y is None else y


# 3️⃣ Feature Extraction - Lag Features

class LagFeatureExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, n_lags=2, target_col='event'):
        self.n_lags = n_lags  # Number of past values to consider
        self.target_col = target_col  # Column to predict, which should be excluded

    def fit(self, X, y=None):
        return self  # No fitting required

    def transform(self, X, y=None):
        X = X.copy()
        X.sort_values(by=['ProcessId', 'DateTime'], inplace=True)

        exclude_cols = {'ProcessId', 'DateTime', self.target_col}
        numeric_cols = [col for col in X.select_dtypes(include=[np.number]).columns if col not in exclude_cols]

        lagged_data = []

        for process_id, group in X.groupby('ProcessId'):
            group = group.sort_values(by='DateTime').copy()

            lagged_features = group[numeric_cols].shift(1).rolling(window=self.n_lags, min_periods=1).mean()

            lagged_features.columns = [f"{col}_lag_mean_{self.n_lags}" for col in lagged_features.columns]

            group = pd.concat([group, lagged_features], axis=1)
            group = group.iloc[self.n_lags:]

            lagged_data.append(group)

        X_transformed = pd.concat(lagged_data).reset_index(drop=True)

        # Return X and y together
        return X_transformed.drop(columns=['event']), X_transformed['event'] if y is None else y


# 4️⃣ Feature Selection - Correlation

class CorrelationFeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(self, threshold=0.9):
        self.threshold = threshold
        self.drop_features = []
        self.protected_columns = ["DateTime", "ProcessId", "event"]

    def fit(self, X, y=None):
        numeric_cols = [col for col in X.select_dtypes(include=[np.number]).columns if col not in self.protected_columns]
        corr_matrix = X[numeric_cols].corr().abs()
        upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

        self.drop_features = [col for col in upper_tri.columns if any(upper_tri[col] > self.threshold)]
        return self

    def transform(self, X, y=None):
        X_transformed = X.drop(columns=self.drop_features, errors="ignore")
        return X_transformed.drop(columns=['event']), X_transformed['event'] if y is None else y


# 5️⃣ Feature Selection - PCA

class PCAFeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(self, variance_threshold=0.95, exclude_cols=None):
        self.variance_threshold = variance_threshold
        self.exclude_cols = exclude_cols if exclude_cols is not None else ['ProcessId', 'DateTime', 'event']
        self.pca = None
        self.num_components = None

    def fit(self, X, y=None):
        feature_cols = [col for col in X.select_dtypes(include=[np.number]).columns if col not in self.exclude_cols]
        X_numeric = X[feature_cols]

        self.pca = PCA()
        self.pca.fit(X_numeric)
        explained_variance_ratio = np.cumsum(self.pca.explained_variance_ratio_)

        self.num_components = np.argmax(explained_variance_ratio >= self.variance_threshold) + 1
        return self

    def transform(self, X, y=None):
        X = X.copy()
        feature_cols = [col for col in X.select_dtypes(include=[np.number]).columns if col not in self.exclude_cols]
        X_numeric = X[feature_cols]

        X_pca = self.pca.transform(X_numeric)[:, :self.num_components]
        pca_cols = [f'PC{i+1}' for i in range(self.num_components)]
        X_pca_df = pd.DataFrame(X_pca, columns=pca_cols, index=X.index)

        X_transformed = pd.concat([X[self.exclude_cols], X_pca_df], axis=1)

        return X_transformed.drop(columns=['event']), X_transformed['event'] if y is None else y
