import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted

from transformers import RegularityResampler, NAInterpolator, LagFeatureExtractor, CorrelationFeatureSelector, PCAFeatureSelector

class MetaClassifier(BaseEstimator, ClassifierMixin):
    """
    A simple meta-classifier that wraps a base estimator.
    
    Parameters
    ----------
    base_estimator : object
        The base estimator to wrap. Must implement fit and predict methods.
    
    Attributes
    ----------
    estimator_ : object
        The fitted base estimator.
    
    classes_ : ndarray, shape (n_classes,)
        The classes labels.
    
    feature_importances_ : ndarray, shape (n_features,)
        The feature importances from the base estimator
        (if the base estimator supports this).
    """
    
    def __init__(self, base_estimator):
        self.base_estimator = base_estimator
    
    def fit(self, X, y):
        """
        Fit the meta-classifier.
        
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The training input samples.
        y : array-like, shape (n_samples,)
            The target values.
        
        Returns
        -------
        self : object
            Returns self.
        """
        # Check input data
        X, y = check_X_y(X, y)
        
        # Store the classes
        self.classes_ = np.unique(y)
        
        # Create a clone of the base estimator
        self.estimator_ = clone(self.base_estimator)
        
        # Fit the estimator on the data
        self.estimator_.fit(X, y)
        
        # Copy feature importances if base estimator supports it
        if hasattr(self.estimator_, 'feature_importances_'):
            self.feature_importances_ = self.estimator_.feature_importances_
        
        return self
    
    def predict(self, X):
        """
        Predict class labels for X.
        
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The input samples.
        
        Returns
        -------
        y_pred : ndarray, shape (n_samples,)
            The predicted class labels.
        """
        # Check if fit has been called
        check_is_fitted(self, ['estimator_', 'classes_'])
        
        # Check input data
        X = check_array(X)
        
        # Predict with base estimator
        return self.estimator_.predict(X)
    
    def predict_proba(self, X):
        """
        Predict class probabilities for X.
        
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The input samples.
        
        Returns
        -------
        proba : ndarray, shape (n_samples, n_classes)
            The class probabilities of the input samples.
        """
        # Check if fit has been called
        check_is_fitted(self, ['estimator_', 'classes_'])
        
        # Check if base estimator implements predict_proba
        if not hasattr(self.estimator_, 'predict_proba'):
            raise AttributeError("Base estimator doesn't implement predict_proba")
        
        # Check input data
        X = check_array(X)
        
        # Get probabilities from estimator
        return self.estimator_.predict_proba(X)

def train_test_split_by_time(df, time_col='DateTime', id_col='ProcessId', train_ratio=0.7):
    """
    Splits the dataset into training and testing sets based on time.
    
    Parameters:
        df (pd.DataFrame): The input dataset.
        time_col (str): The name of the datetime column.
        id_col (str): The process identifier column.
        train_ratio (float): Proportion of oldest timestamps to use for training.
    
    Returns:
        train_df (pd.DataFrame): Training set (oldest 70% of timestamps per ProcessId).
        test_df (pd.DataFrame): Testing set (newest 30% of timestamps per ProcessId).
    """
    train_list = []
    test_list = []

    for process_id, group in df.groupby(id_col):
        group = group.sort_values(by=time_col)  # Ensure chronological order
        split_idx = int(len(group) * train_ratio)  # Determine the split index
        train_list.append(group.iloc[:split_idx])  # Oldest 70% for training
        test_list.append(group.iloc[split_idx:])  # Newest 30% for testing

    train_df = pd.concat(train_list).reset_index(drop=True)
    test_df = pd.concat(test_list).reset_index(drop=True)

    return train_df, test_df

# Example usage
if __name__ == "__main__":

    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score
    
    # Load raw dataset
    raw_df = pd.read_pickle("Datasets/final_dataset.pkl")

    # Split into train & test
    train_df, test_df = train_test_split_by_time(raw_df, train_ratio=0.7)

    X_train, y_train = train_df.drop(columns=['event']), train_df['event']
    X_test, y_test = test_df.drop(columns=['event']), test_df['event']

    param_dist = {
    # Regularity Resampler hyperparameters
    'regularity_resampling__freq': ['15T', '30T', '1H', '2H', '4H'],  # Frequencies for resampling

    # NAInterpolator hyperparameters
    'imputation__method': ['linear', 'polynomial', 'spline'],  # Different interpolation methods
    'imputation__event_fill': [0, 1],  # Event fill values (0 or 1)

    # LagFeatureExtractor hyperparameters
    'feature_extraction__n_lags': [1, 2, 3, 4, 5],  # Number of lags for feature extraction

    # CorrelationFeatureSelector hyperparameters
    'feature_selection__threshold': [0.85, 0.9, 0.95],  # Correlation thresholds for feature selection
}

    # Create a random forest classifier as the base estimator
    base_clf = RandomForestClassifier(n_estimators=10, random_state=42)
    
    # Create the meta-classifier
    meta_clf = MetaClassifier(base_estimator=base_clf)

    # Example pipeline (same as before, just adding your transformers)
    pipeline = Pipeline([
        ('regularity_resampling', RegularityResampler()),
        ('imputation', NAInterpolator()),
        ('feature_extraction', LagFeatureExtractor()),
        ('feature_selection', CorrelationFeatureSelector()),
        ('classifier', meta_clf)  # Replace with any other model you prefer
    ])
    
    random_search = RandomizedSearchCV(
        pipeline, param_distributions=param_dist, n_iter=10, cv=3, n_jobs=-1
    )

    random_search.fit(train_df, y_train)
    
