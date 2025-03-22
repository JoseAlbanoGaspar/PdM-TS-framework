import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from itertools import product

from transformers import (
    RegularityResampler,
    NAForwardFill,
    LagFeatureExtractor,
    CorrelationFeatureSelector
)
    
class MetaClassifier(BaseEstimator, ClassifierMixin):
    """
    A meta-classifier that wraps a base estimator.
    Ensures compatibility with pipelines that transform both X and y.
    """
    
    def __init__(self, base_estimator):
        self.base_estimator = base_estimator
    
    def fit(self, X, _y=None):
        """
        Fit the meta-classifier.
        X should contain both features and target (before splitting).
        """
        y = X['event']  # Extract target
        X = X.drop(columns=['event'])  # Remove target from features
        X = self.prepare_datetime_data(X)  # Process datetime features

        X, y = check_X_y(X, y)  # Validate inputs
        self.classes_ = np.unique(y)

        self.estimator_ = clone(self.base_estimator)  # Clone model
        self.estimator_.fit(X, y)  # Train model

        if hasattr(self.estimator_, 'feature_importances_'):
            self.feature_importances_ = self.estimator_.feature_importances_

        return self
    
    def predict(self, X):
        """
        Predict class labels for X.
        """
        check_is_fitted(self, ['estimator_', 'classes_'])

        if 'event' in X.columns:  # Drop event if it exists
            X = X.drop(columns=['event'])
        X = self.prepare_datetime_data(X)

        X = check_array(X)
        return self.estimator_.predict(X)

    def predict_proba(self, X):
        """
        Predict class probabilities.
        """
        check_is_fitted(self, ['estimator_', 'classes_'])

        if not hasattr(self.estimator_, 'predict_proba'):
            raise AttributeError("Base estimator doesn't support predict_proba")
        
        if 'event' in X.columns:  # Drop event if present
            X = X.drop(columns=['event'])
        X = self.prepare_datetime_data(X)

        X = check_array(X)
        return self.estimator_.predict_proba(X)
    
    def prepare_datetime_data(self, df):
        """
        Process datetime features.
        """
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


class CustomRandomizedSearchCV(RandomizedSearchCV):
    """Custom RandomizedSearchCV that uses transformed y as ground truth"""
    
    def fit(self, X, y=None, groups=None, **fit_params):
        """Override fit method to handle transformed y values"""
        # Initialize results dictionary
        self.cv_results_ = {
            'mean_test_score': [],
            'params': []
        }
        self.best_score_ = -np.inf
        self.best_params_ = None
        self.best_index_ = None
        
        # Generate all possible parameter combinations
        param_names = sorted(self.param_distributions)
        param_values = [self.param_distributions[name] for name in param_names]
        param_combinations = [dict(zip(param_names, v)) for v in product(*param_values)]
        
        # For each parameter combination
        for parameters in param_combinations:
            print(f"Testing parameters: {parameters}", flush=True)
            # Clone the pipeline
            estimator = clone(self.estimator)
            estimator.set_params(**parameters)
            
            scores = []
            # For each CV split
            for train_idx, val_idx in self.cv.split(X, y, groups):
                # Split data
                X_train = X.iloc[train_idx].copy()
                X_val = X.iloc[val_idx].copy()
                
                # Fit pipeline on training data
                estimator.fit(X_train)
                
                # Get transformed validation data and ground truth
                X_val_transformed = X_val.copy()
                for name, transformer in estimator.steps[:-1]:
                    if hasattr(transformer, 'transform'):
                        X_val_transformed = transformer.transform(X_val_transformed)
                
                # Extract transformed y as ground truth
                y_true = X_val_transformed['event']
                
                # Get predictions
                y_pred = estimator.predict(X_val)
                
                # Calculate score
                score = accuracy_score(y_true, y_pred)
                scores.append(score)
            
            # Store results
            mean_score = np.mean(scores)
            self.cv_results_['mean_test_score'].append(mean_score)
            self.cv_results_['params'].append(parameters)
            
            # Update best score/parameters if needed
            if mean_score > self.best_score_:
                self.best_score_ = mean_score
                self.best_params_ = parameters
                self.best_index_ = len(self.cv_results_['mean_test_score']) - 1
                self.best_estimator_ = clone(estimator)
        
        # Fit the best estimator on the full dataset
        if self.refit:
            self.best_estimator_.fit(X)
        
        return self

# ...existing MetaClassifier and train_test_split_by_time code...

if __name__ == "__main__":
    # Load dataset
    raw_df = pd.read_pickle("Datasets/final_dataset.pkl")
    
    # Split dataset into train & test
    train_df, test_df = train_test_split_by_time(raw_df, train_ratio=0.7)

    param_dist = {
        'regularity_resampling__freq': ['15T', '30T', '1H', '2H', '4H'],
        'feature_extraction__n_lags': [1, 2, 3, 4, 5],
        'feature_selection__threshold': [0.85, 0.9, 0.95],
    }

    base_clf = RandomForestClassifier(n_estimators=10, random_state=42)
    meta_clf = MetaClassifier(base_estimator=base_clf)

    # Pipeline with transformers
    pipeline = Pipeline([
        ('regularity_resampling', RegularityResampler()),
        ('imputation', NAForwardFill()),
        ('feature_extraction', LagFeatureExtractor()),
        ('feature_selection', CorrelationFeatureSelector()),
        ('classifier', meta_clf) 
    ])

    # Time Series Cross-Validation
    tscv = TimeSeriesSplit(n_splits=2)

    # Use CustomRandomizedSearchCV
    random_search = CustomRandomizedSearchCV(
        pipeline, 
        param_distributions=param_dist,
        n_iter=10,
        cv=tscv,
        n_jobs=1,  # Set to 1 since we're handling CV manually
        verbose=2
    )

    # Train the model using complete dataframe
    random_search.fit(train_df, train_df['event'])

    # Get transformed test data for final evaluation
    test_transformed = test_df.copy()
    for name, transformer in random_search.best_estimator_.steps[:-1]:
        if hasattr(transformer, 'transform'):
            test_transformed = transformer.transform(test_transformed)
    
    # Evaluate performance using transformed labels
    y_pred = random_search.best_estimator_.predict(test_df)
    y_true = test_transformed['event']
    print("Accuracy:", accuracy_score(y_true, y_pred))
    print("Best Parameters:", random_search.best_params_)