import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import precision_score, recall_score, confusion_matrix
from FE_configuration import feature_extraction_preprocessing
from transformers import ImputationWrapper, FeatureExtractorWrapper, FeatureSelectionWrapper
from utils import train_test_split_by_time
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import roc_auc_score
# Load dataset
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
    


# Load dataset
raw_df = pd.read_pickle("Datasets/final_dataset.pkl")

# Feature extraction configuration
N_FEATURES = 10
N_TSFEL_FEATURES = 10
N_TSFRESH_FEATURES = 10
# Dataset column configuration
COLUMN_CONFIG = {
    'primary_key': ['ProcessId', 'DateTime'],
    'time_col': 'DateTime',
    'target_col': 'event',
    'protected_cols': ['ProcessId', 'DateTime', 'event'],
    'id_col': 'ProcessId',
}
# Split dataset into train & test
train_df, test_df = train_test_split_by_time(raw_df, train_ratio=0.7)

# Feature extraction preprocessing (use a small subset for speed)
sub_train_df, _ = train_test_split_by_time(raw_df, train_ratio=0.1)
tsfel_config_file, top_features, tsfresh_fc_parameters = feature_extraction_preprocessing(
    sub_train_df, COLUMN_CONFIG, N_FEATURES, N_TSFEL_FEATURES, N_TSFRESH_FEATURES
)

# Best parameters from search
best_params = {
    "classifier__base_estimator__max_depth": 5,
    "classifier__base_estimator__n_estimators": 100,
    "feature_extraction__n_lags": 5,
    "feature_extraction__params": ('pycatch22', {}),
    "feature_selection__strategy": "correlation",
    "feature_selection__threshold": 0.85,
    "imputation__params": ('interpolate', 'linear')
}

# Build pipeline with best parameters
meta_clf = MetaClassifier(
    base_estimator=LGBMClassifier(
        max_depth=best_params["classifier__base_estimator__max_depth"],
        n_estimators=best_params["classifier__base_estimator__n_estimators"]
    ),
    column_config=COLUMN_CONFIG
)

pipeline = Pipeline([
    ('imputation', ImputationWrapper(params=best_params["imputation__params"], column_config=COLUMN_CONFIG)),
    ('feature_extraction', FeatureExtractorWrapper(
        params=best_params["feature_extraction__params"],
        n_lags=best_params["feature_extraction__n_lags"],
        column_config=COLUMN_CONFIG,
        features=top_features
    )),
    ('feature_selection', FeatureSelectionWrapper(
        strategy=best_params["feature_selection__strategy"],
        threshold=best_params["feature_selection__threshold"],
        column_config=COLUMN_CONFIG
    )),
])

# Fit and transform train data
X_train_transformed = pipeline.fit_transform(train_df)
y_train = train_df[COLUMN_CONFIG['target_col']]

# Transform test data
X_test_transformed = pipeline.transform(test_df)
y_test = test_df[COLUMN_CONFIG['target_col']]

# For MetaClassifier, X must include protected columns and target_col for encoding
X_train_for_meta = train_df.copy()
meta_clf.fit(X_train_for_meta)

# Predict on test data
y_pred = meta_clf.predict(test_df)
y_proba = meta_clf.predict_proba(test_df)[:, 1]

# Compute metrics
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
auc = roc_auc_score(y_test, y_proba)

print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"AUC: {auc:.4f}")
print(f"False Positives: {fp}")
print(f"False Negatives: {fn}")