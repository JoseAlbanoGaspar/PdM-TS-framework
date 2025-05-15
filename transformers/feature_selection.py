from sklearn.decomposition import PCA
from utils import DEFAULT_COLUMN_CONFIG
from sklearn.base import BaseEstimator, TransformerMixin

import pandas as pd
import numpy as np

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
        self.feature_names = None  # To store feature names after transformation
    
    def get_feature_names(self):
        """Returns the names of the features after transformation."""
        if not hasattr(self, 'feature_names'):
            raise AttributeError("Feature names are only available after transforming the transformer.")
        return self.feature_names
    
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
        
        ret_value = self._selector.transform(X)
        self.feature_names = ret_value.columns.tolist()
        return ret_value
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