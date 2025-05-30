from utils import DEFAULT_COLUMN_CONFIG
from sklearn.base import BaseEstimator, TransformerMixin

import pandas as pd
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
            process_func=lambda df: df.interpolate(**kwargs).ffill().bfill(),
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
            process_func=lambda df: df.ffill().bfill(),
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
