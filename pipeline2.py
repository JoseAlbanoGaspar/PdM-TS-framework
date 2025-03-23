import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.datasets import make_classification
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.model_selection._split import BaseCrossValidator
from sklearn.metrics import accuracy_score

from transformers import (
    RegularityResampler,
    NAInterpolator,
    LagFeatureExtractor,
    CorrelationFeatureSelector
)

# meta-modelo que recebe o modelo e ignora um y_dummy
# a func score é importante
class MetaClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, base_estimator, actual_target_col='event'):
        self.base_estimator = base_estimator
        self.actual_target_col = actual_target_col

    def fit(self, X, y=None):
        y_actual = X[self.actual_target_col]
        X_actual = X.drop(columns=[self.actual_target_col, 'DateTime'])
        self.base_estimator.fit(X_actual, y_actual)
        return self

    def predict(self, X):
        X_actual = X.drop(columns=[self.actual_target_col, 'DateTime'])
        return self.base_estimator.predict(X_actual)

    def predict_proba(self, X):
        X_actual = X.drop(columns=[self.actual_target_col, 'DateTime'])
        return self.base_estimator.predict_proba(X_actual)

    def score(self, X, y=None):
        y_true = X[self.actual_target_col]

        y_pred = self.predict(X)

        return accuracy_score(y_true, y_pred)

def train_test_split_by_time(df, time_col='DateTime', id_col='ProcessId', train_ratio=0.7):
    """
    Splits the dataset into training and testing sets based on time.
    """
    train_list, test_list = [], []

    for process_id, group in df.groupby(id_col):
        group = group.sort_values(by=time_col)
        split_idx = int(len(group) * train_ratio)
        train_list.append(group.iloc[:split_idx])
        test_list.append(group.iloc[split_idx:])

    train_df = pd.concat(train_list).reset_index(drop=True)
    test_df = pd.concat(test_list).reset_index(drop=True)

    return train_df, test_df


# Load dataset
raw_df = pd.read_pickle("Datasets/final_dataset.pkl")
#raw_df = raw_df.head(20)
# Split dataset into train & test
train_df, test_df = train_test_split_by_time(raw_df, train_ratio=0.7)

X_train, y_train = train_df.drop(columns=['event']), train_df['event']
X_test, y_test = test_df.drop(columns=['event']), test_df['event']


meta_clf = MetaClassifier(
        base_estimator=DecisionTreeClassifier(),
        actual_target_col='event'
    )
# pipeline simples
pipeline = Pipeline([
        ('regularity_resampling', RegularityResampler()),
        ('imputation', NAInterpolator()),
        ('feature_extraction', LagFeatureExtractor()),
        ('feature_selection', CorrelationFeatureSelector()),
        ('classifier', meta_clf) 
    ])

param_distributions = {
    'regularity_resampling__freq': ['15T', '30T', '1H', '2H', '4H'],
    'imputation__method': ['linear', ('polynomial', 2), ('polynomial', 3), ('spline', 2), ('spline', 3)],
    'feature_extraction__n_lags': [1, 2, 3, 4, 5],
    'feature_selection__threshold': [0.85, 0.9, 0.95],
    'classifier__base_estimator__max_depth': [3, 5, 10, None]
}

random_search = RandomizedSearchCV(
    pipeline,
    param_distributions,
    cv=TimeSeriesSplit(n_splits=2),
    n_iter=5,
)

# Fit with original data
random_search.fit(train_df, y_train) # y_train is a dummy
print(random_search.best_params_)
print(random_search.best_score_)


# getting accuracy
print("Accuracy:", random_search.best_estimator_.score(test_df))
print("Best Params:", random_search.best_params_)
