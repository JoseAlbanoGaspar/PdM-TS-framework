import os
import json
import pandas as pd
import lightgbm as lgb
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin

from transformers import (
    RegularityResampler, NAInterpolator, LagFeatureExtractor,
    CorrelationFeatureSelector, PCAFeatureSelector
)

# Wrapper for Transformers
class StepSelector(BaseEstimator, TransformerMixin):
    def __init__(self, transformer=None, params={}):
        self.transformer = transformer
        self.params = params
    
    def fit(self, X, y=None):
        if self.transformer:
            self.transformer.set_params(**self.params)
            self.transformer.fit(X, y)
        return self
    
    def transform(self, X, y=None):
        if self.transformer:
            return self.transformer.transform(X, y)
        return X, y  # Ensure we always return (X, y)

# Custom Pipeline with y handling
class PipelineWithTarget(Pipeline):
    def fit(self, X, y=None, **fit_params):
        """Ensures 'event' (y) stays with X throughout transformations."""
        X, y = X.copy(), y.copy()
        
        for name, transformer in self.steps[:-1]:  
            print(f"▶ Applying {name}")
            X, y = transformer.fit_transform(X, y)
            print(X.head())  # Debugging Output
        
        return self.steps[-1][1].fit(X, y, **fit_params)  # Fit Model

    def transform(self, X, y=None):
        X, y = X.copy(), (y.copy() if y is not None else None)

        for name, transformer in self.steps[:-1]:
            X, y = transformer.transform(X, y)

        return X, y  # Return transformed X and y

# Load dataset
raw_df = pd.read_pickle("Datasets/final_dataset.pkl")

# Train-test split
train_ratio = 0.7
train_df = raw_df.iloc[:int(len(raw_df) * train_ratio)].reset_index(drop=True)
test_df = raw_df.iloc[int(len(raw_df) * train_ratio):].reset_index(drop=True)

# Extract X and y
X_train, X_test = train_df.drop(columns=['event']), test_df.drop(columns=['event'])
y_train, y_test = train_df['event'], test_df['event']

# Define Pipeline
pipeline = PipelineWithTarget([
    ('regularity', StepSelector(RegularityResampler(), {'freq': '1H'})),
    ('imputation', StepSelector(NAInterpolator(), {'method': 'linear'})),
    ('feature_extraction', StepSelector(LagFeatureExtractor(), {'n_lags': 3})),
    ('feature_selection', StepSelector(CorrelationFeatureSelector(), {'threshold': 0.9})),
    ('classifier', lgb.LGBMClassifier(n_estimators=100, max_depth=10, learning_rate=0.05, num_leaves=31, random_state=42))
])

# 🚀 Train Pipeline
pipeline.fit(X_train, y_train)

# Evaluate on Test Set
X_test_transformed, y_test_transformed = pipeline.transform(X_test, y_test)
accuracy = pipeline.steps[-1][1].score(X_test_transformed, y_test_transformed)

print(f"✅ Model Accuracy: {accuracy:.4f}")
