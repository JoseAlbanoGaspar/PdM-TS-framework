from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MinMaxScaler, StandardScaler

class NormalizationWrapper(BaseEstimator, TransformerMixin):
    """
    A wrapper for MinMaxScaler and StandardScaler normalization methods.
    
    Parameters:
    -----------
    scaler_method : str
        The scaling method to use. Options:
        - 'minmax': scales to range [0,1]
        - 'standard': standardizes features to mean=0 and variance=1
    column_config : dict
        Configuration specifying protected columns
    """
    def __init__(self, scaler_method='minmax', column_config=None):
        self.scaler_method = scaler_method
        self.column_config = column_config
        self.protected_cols = column_config['protected_cols']
        
        # Initialize scaler based on method
        if scaler_method == 'minmax':
            self.scaler = MinMaxScaler()  # default range [0,1]
        elif scaler_method == 'standard':
            self.scaler = StandardScaler()  # default mean=0, std=1
        else:
            raise ValueError("Scaler method must be either 'minmax' or 'standard'")
        
    def fit(self, X, y=None):
        self.feature_cols = [col for col in X.columns if col not in self.protected_cols]
        self.scaler.fit(X[self.feature_cols])
        return self
        
    def transform(self, X):
        X_transformed = X.copy()
        X_transformed[self.feature_cols] = self.scaler.transform(X[self.feature_cols])
        return X_transformed