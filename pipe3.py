import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted

# Import custom transformers (assuming you have them implemented)
from transformers import RegularityResampler, NAForwardFill, LagFeatureExtractor, CorrelationFeatureSelector

class MetaClassifier(BaseEstimator, ClassifierMixin):
    """
    A simple meta-classifier that wraps a base estimator.
    Automatically handles separation of 'event' if it appears in X.
    """

    def __init__(self, base_estimator):
        self.base_estimator = base_estimator

    def fit(self, X, y=None):
        """Fit the classifier. Automatically extracts 'event' from X if present."""
        if 'event' in X.columns:
            y = X['event']
            X = X.drop(columns=['event'])

        X = self.prepare_datetime_data(X)

        X, y = check_X_y(X, y)

        self.classes_ = np.unique(y)
        self.estimator_ = clone(self.base_estimator)
        self.estimator_.fit(X, y)

        return self

    def predict(self, X):
        """Predict class labels. Automatically removes 'event' if present."""
        check_is_fitted(self, ['estimator_', 'classes_'])

        if 'event' in X.columns:
            X = X.drop(columns=['event'])

        X = self.prepare_datetime_data(X)
        X = check_array(X)

        return self.estimator_.predict(X)

    def predict_proba(self, X):
        """Predict class probabilities. Automatically removes 'event' if present."""
        check_is_fitted(self, ['estimator_', 'classes_'])

        if not hasattr(self.estimator_, 'predict_proba'):
            raise AttributeError("Base estimator doesn't implement predict_proba")

        if 'event' in X.columns:
            X = X.drop(columns=['event'])

        X = self.prepare_datetime_data(X)
        X = check_array(X)

        return self.estimator_.predict_proba(X)

    def prepare_datetime_data(self, df):
        """Extract datetime features dynamically."""
        df = df.copy()
        if 'DateTime' in df.columns:
            df['Year'] = df['DateTime'].dt.year
            df['Month'] = df['DateTime'].dt.month
            df['Day'] = df['DateTime'].dt.day
            df['Hour'] = df['DateTime'].dt.hour
            df['Minute'] = df['DateTime'].dt.minute
            df.drop(columns=['DateTime'], inplace=True)
        return df


def train_test_split_by_time(df, time_col='DateTime', id_col='ProcessId', train_ratio=0.7):
    """
    Splits the dataset into training and testing sets based on time.
    Ensures that each ProcessId is split correctly.
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


if __name__ == "__main__":
    # Load dataset
    raw_df = pd.read_pickle("Datasets/final_dataset.pkl")

    # Split into train & test
    train_df, test_df = train_test_split_by_time(raw_df, train_ratio=0.7)

    # Extract X, y
    X_train, y_train = train_df.drop(columns=['event']), train_df['event']
    X_test, y_test = test_df.drop(columns=['event']), test_df['event']

    # Define parameter grid
    param_dist = {
        'regularity_resampling__freq': ['15T', '30T', '1H', '2H', '4H'],
        #'imputation__method': ['linear', 'polynomial', 'spline'],
        'imputation__event_fill': [0, 1],
        'feature_extraction__n_lags': [1, 2, 3, 4, 5],
        'feature_selection__threshold': [0.85, 0.9, 0.95],
    }

    # Base classifier
    base_clf = RandomForestClassifier(n_estimators=10, random_state=42)
    
    # Meta-classifier
    meta_clf = MetaClassifier(base_estimator=base_clf)

    # Define pipeline
    pipeline = Pipeline([
        ('regularity_resampling', RegularityResampler()),
        ('imputation', NAForwardFill()), 
        ('feature_extraction', LagFeatureExtractor()),
        ('feature_selection', CorrelationFeatureSelector()), 
        ('classifier', meta_clf)
    ])
    
    # Perform RandomizedSearchCV
    random_search = RandomizedSearchCV(
        pipeline, param_distributions=param_dist, n_iter=10, cv=3, n_jobs=-1
    )

    random_search.fit(train_df, y_train)  # ✅ Now works properly
