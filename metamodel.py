from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import roc_auc_score

class MetaClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, base_estimator, column_config=None):
        self.base_estimator = base_estimator
        self.column_config = column_config
        self.column_mapping = None
        
    def _encode_column_names(self, X):
        # Create mapping for problematic column names
        self.column_mapping = {col: f'col_{i}' for i, col in enumerate(X.columns)}
        return X.rename(columns=self.column_mapping)
    
    def _decode_column_names(self, X):
        # Reverse the mapping
        reverse_mapping = {v: k for k, v in self.column_mapping.items()}
        return X.rename(columns=reverse_mapping)

    def fit(self, X, y=None):
        y_actual = X[self.column_config['target_col']]
        X_actual = X.drop(columns=self.column_config['protected_cols'])
        
        # Encode column names before training
        X_encoded = self._encode_column_names(X_actual)
        self.base_estimator.fit(X_encoded, y_actual)
        return self

    def predict(self, X):
        X_actual = X.drop(columns=self.column_config['protected_cols'])
        # Encode column names for prediction
        X_encoded = X_actual.rename(columns=self.column_mapping)
        return self.base_estimator.predict(X_encoded)

    def predict_proba(self, X):
        X_actual = X.drop(columns=self.column_config['protected_cols'])
        # Encode column names for prediction
        X_encoded = X_actual.rename(columns=self.column_mapping)
        return self.base_estimator.predict_proba(X_encoded)

    def score(self, X, y=None):
        y_true = X[self.column_config['target_col']]
        y_pred = self.predict_proba(X)
        return roc_auc_score(y_true, y_pred[:, 1])
    


