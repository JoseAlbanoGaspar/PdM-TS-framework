import os
import json
import pandas as pd
import lightgbm as lgb
from itertools import product
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.base import BaseEstimator, TransformerMixin

from transformers import (
    RegularityResampler, NAInterpolator, LagFeatureExtractor,
    CorrelationFeatureSelector, PCAFeatureSelector
)


# Wrapper Transformer for Grid Search
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
        else:
            return X


# Custom pipeline wrapper to handle both X and y
class PipelineWithTarget(Pipeline):
    def fit(self, X, y=None, **fit_params):
        """Ensures that event (y) stays with X until transformation is complete."""

        X['event'] = y.values  # Temporarily store y inside X
        print(X['event'].head())
        X_transformed = X
        for name, transformer in self.steps[:-1]:  # Apply all transformers
            X_transformed, y = transformer.fit_transform(X_transformed, X_transformed['event'])
            print(X_transformed['event'].head())


        X_final = X_transformed.drop(columns=['event'])  # Extract final X
        y_final = X_transformed['event']  # Extract updated y

        return self.steps[-1][1].fit(X_final, y_final, **fit_params)  # Fit model

    def transform(self, X, y=None):
        X = X.copy()
        X['event'] = y.values if y is not None else None  # Keep event inside X
        
        for name, transformer in self.steps[:-1]:
            X, y = transformer.transform(X, X['event'])

        return X.drop(columns=['event']), X['event']  # Return updated X and y


# Train-test split by time
def train_test_split_by_time(df, time_col='DateTime', id_col='ProcessId', train_ratio=0.7):
    train_list, test_list = [], []
    for process_id, group in df.groupby(id_col):
        group = group.sort_values(by=time_col)
        split_idx = int(len(group) * train_ratio)
        train_list.append(group.iloc[:split_idx])
        test_list.append(group.iloc[split_idx:])
    return pd.concat(train_list).reset_index(drop=True), pd.concat(test_list).reset_index(drop=True)


# Load dataset
raw_df = pd.read_pickle("Datasets/final_dataset.pkl")
train_df, test_df = train_test_split_by_time(raw_df, train_ratio=0.7)

# Define pipeline steps with multiple options
param_grid = {
    'regularity': [None, StepSelector(RegularityResampler(), {'freq': '1H'}), StepSelector(RegularityResampler(), {'freq': '2H'})],
    'imputation': [StepSelector(NAInterpolator(), {'method': 'linear'})],
    'feature_extraction': [StepSelector(LagFeatureExtractor(), {'n_lags': 3}), StepSelector(LagFeatureExtractor(), {'n_lags': 6})],
    'feature_selection': [StepSelector(CorrelationFeatureSelector(), {'threshold': 0.9}), StepSelector(PCAFeatureSelector(), {'variance_threshold': 0.95})],
}

# LightGBM model
model = lgb.LGBMClassifier(random_state=42)

# Define Pipeline with custom handling of X and y
pipeline = PipelineWithTarget([
    ('regularity', StepSelector()),
    ('imputation', StepSelector()),
    ('feature_extraction', StepSelector()),
    ('feature_selection', StepSelector()),
    ('classifier', model)
])

# Flatten param_grid for RandomizedSearchCV
param_search = {
    'regularity': param_grid['regularity'],
    'imputation': param_grid['imputation'],
    'feature_extraction': param_grid['feature_extraction'],
    'feature_selection': param_grid['feature_selection'],
    'classifier__n_estimators': [50, 100, 200],
    'classifier__max_depth': [10, 20, 30, None],
    'classifier__learning_rate': [0.01, 0.05, 0.1],
    'classifier__num_leaves': [31, 50, 100],
}

# Train-test split for model training
X_train, X_test, y_train, y_test = train_df.drop(columns=['event']), test_df.drop(columns=['event']), train_df['event'], test_df['event']

# INSERT HERE
'''
# Run RandomizedSearchCV
random_search = RandomizedSearchCV(pipeline, param_distributions=param_search, n_iter=30, cv=3, scoring='accuracy', verbose=1, n_jobs=-1, random_state=42)
random_search.fit(train_df, y_train)

# Save best model and metadata
best_model = random_search.best_estimator_
metadata = {'best_params': random_search.best_params_}
output_dir = "DatasetCleaned"
os.makedirs(output_dir, exist_ok=True)
with open(os.path.join(output_dir, "best_model.pkl"), "wb") as f:
    json.dump(metadata, f, indent=4)

print(f"✅ Best model saved! Accuracy: {best_model.score(X_test, y_test):.4f}")
'''