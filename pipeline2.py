import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.ensemble import RandomForestClassifier

from transformers import (
    RegularityResampler, NAInterpolator, LagFeatureExtractor,
    CorrelationFeatureSelector, PCAFeatureSelector
)

# 1. Wrapper to Track y Transformations
class TransformerYTracker(BaseEstimator, TransformerMixin):
    def __init__(self, transformer):
        self.transformer = transformer
        self.y_transformed_ = None
        
    def fit(self, X, y=None):
        self.transformer.fit(X, y)
        return self
    
    def transform(self, X, y=None):
        X_transformed = self.transformer.transform(X)
        
        if hasattr(self.transformer, 'y_new_'):
            self.y_transformed_ = self.transformer.y_new_
        else:
            self.y_transformed_ = y
        
        return X_transformed
        
    def get_y_transformed(self):
        return self.y_transformed_

# 2. MetaEstimator to Manage Full Pipeline
class MetaEstimator(BaseEstimator):
    def __init__(self, transformers, final_estimator):
        self.transformers = transformers
        self.final_estimator = final_estimator
        self._transformers = None
    
    def fit(self, X, y):
        X_current = X.copy()
        y_current = y.copy()
        
        self._transformers = []
        
        for transformer in self.transformers:
            wrapped_transformer = TransformerYTracker(clone(transformer))
            X_current = wrapped_transformer.fit_transform(X_current, y_current)
            y_current = wrapped_transformer.get_y_transformed()
            self._transformers.append(wrapped_transformer)
        
        self.final_estimator.fit(X_current, y_current)
        return self
    
    def predict(self, X):
        X_current = X.copy()
        for transformer in self._transformers:
            X_current = transformer.transform(X_current)
        return self.final_estimator.predict(X_current)
    
    def get_params(self, deep=True):
        params = {}
        for i, t in enumerate(self.transformers):
            for k, v in t.get_params(deep=deep).items():
                params[f'transformers__{i}__{k}'] = v
        
        for k, v in self.final_estimator.get_params(deep=deep).items():
            params[f'final_estimator__{k}'] = v
        
        return params
    
    def set_params(self, **params):
        transformer_params = {}
        estimator_params = {}
        
        for k, v in params.items():
            if k.startswith('transformers__'):
                parts = k.split('__')
                if len(parts) >= 3:
                    idx = int(parts[1])
                    param_name = '__'.join(parts[2:])
                    if idx not in transformer_params:
                        transformer_params[idx] = {}
                    transformer_params[idx][param_name] = v
            elif k.startswith('final_estimator__'):
                param_name = '__'.join(k.split('__')[1:])
                estimator_params[param_name] = v
        
        for idx, params in transformer_params.items():
            self.transformers[idx].set_params(**params)
        
        self.final_estimator.set_params(**estimator_params)
        
        return self

# 3. Define Transformers
resampler = RegularityResampler(freq='1H')
imputer = NAInterpolator(method='linear')
lag_extractor = LagFeatureExtractor(n_lags=3)
corr_selector = CorrelationFeatureSelector(threshold=0.9)

# 4. Define Final Estimator
final_model = RandomForestClassifier(random_state=42)

# 5. Create MetaEstimator
meta_estimator = MetaEstimator(
    transformers=[resampler, imputer, lag_extractor, corr_selector],
    final_estimator=final_model
)

# 6. Setup TimeSeries Cross-Validation
tscv = TimeSeriesSplit(n_splits=5)

# 7. Define Hyperparameter Search Space
param_distributions = {
    'transformers__0__freq': ['30T', '1H', '2H'],
    'transformers__1__method': ['linear', 'nearest', 'spline'],
    'transformers__2__n_lags': [2, 3, 5],
    'transformers__3__threshold': [0.8, 0.9, 0.95],
    'final_estimator__n_estimators': [50, 100, 200],
    'final_estimator__max_depth': [None, 10, 20]
}

# 8. Run Randomized Search
random_search = RandomizedSearchCV(
    estimator=meta_estimator,
    param_distributions=param_distributions,
    n_iter=20,
    cv=tscv,
    scoring='accuracy',
    verbose=2,
    random_state=42,
    n_jobs=-1
)

# Uncomment when ready to run
# random_search.fit(X_train, y_train)
# best_model = random_search.best_estimator_
# print("Best Parameters:", random_search.best_params_)
