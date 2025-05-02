from lightgbm import LGBMClassifier
import numpy as np
import pandas as pd
from sklearn.experimental import enable_halving_search_cv
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.pipeline import Pipeline
from sklearn.model_selection import (
    RandomizedSearchCV,
    GridSearchCV,
    HalvingGridSearchCV,
    HalvingRandomSearchCV,
    TimeSeriesSplit
)
from sklearn.metrics import roc_auc_score
import time  # Import time module for timing execution

from FE_configuration import feature_extraction_preprocessing
from transformers import (
    ImputationWrapper,
    FeatureExtractorWrapper,
    FeatureSelectionWrapper
)

from utils import train_test_split_by_time
import json

# Meta-modelo that receives the model and ignores a y_dummy
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

X_train, y_train = train_df.drop(columns=['event']), train_df['event']
X_test, y_test = test_df.drop(columns=['event']), test_df['event']

# shorten the dataset for faster feature extraction preprocessing
sub_train_df, _ = train_test_split_by_time(raw_df, train_ratio=0.1)

tsfel_config_file, top_features, tsfresh_fc_parameters = feature_extraction_preprocessing(sub_train_df, COLUMN_CONFIG, N_FEATURES, N_TSFEL_FEATURES, N_TSFRESH_FEATURES)

# Create a meta-classifier with decision tree
meta_clf = MetaClassifier(
    base_estimator=LGBMClassifier(),
    column_config=COLUMN_CONFIG
)

pipeline = Pipeline([
    ('imputation', ImputationWrapper(params=('interpolate', 'linear'), column_config=COLUMN_CONFIG)),
    ('feature_extraction', FeatureExtractorWrapper(
        params=('tsfel', {'config_file': tsfel_config_file}),
        n_lags=8,
        column_config=COLUMN_CONFIG,
        features=top_features
    )),
    ('feature_selection', FeatureSelectionWrapper(strategy='correlation', column_config=COLUMN_CONFIG)),
    ('classifier', meta_clf)
])

imputation_params = [
    ('interpolate', 'linear'),
    ('interpolate', ('polynomial', 2)),
    ('interpolate', ('polynomial', 3)),
    ('interpolate', ('spline', 2)),
    ('interpolate', ('spline', 3)),
    ('ffill', None),
]
'''
feature_extraction_params = [
    ('tsfel', {
        'config_file': tsfel_config_file,
    }),
    ('tsfresh', {
        'default_fc_parameters': tsfresh_fc_parameters,
    }),
    ('pycatch22', {
        #'pycatch22_features': pycatch22_features ,   # not implemented on the preprocessing cause pycatch is fast
    })
]'''

feature_extraction_params = [
    
    ('pycatch22', {
        #'pycatch22_features': pycatch22_features ,   # not implemented on the preprocessing cause pycatch is fast
    })
]

param_distributions = {    
    # Imputation parameters
    'imputation__params': imputation_params,

    # Feature extraction parameters
    #'feature_extraction__domains': ['temporal', 'frequency', 'statistical'],
    'feature_extraction__n_lags': [3, 5, 8],
    'feature_extraction__params': feature_extraction_params,

    # Feature selection parameters
    'feature_selection__strategy': ['correlation', 'pca'],
    'feature_selection__threshold': [0.85, 0.9, 0.95, 0.99], # this are thresholds for correlation and pca - works for both
    
    # Classifier parameters
    'classifier__base_estimator__max_depth': [3, 5, 7, 11],
    'classifier__base_estimator__n_estimators': [50, 100, 200, 300]
    }

# Define the cross-validation strategy
cv = TimeSeriesSplit(n_splits=5)
# Configure search_strategy
# 1. Grid Search - tries all possible combinations
grid_search = GridSearchCV(
    pipeline,
    param_distributions,
    cv=cv,
    verbose=1,
    return_train_score=True,
    n_jobs=4,   # Reduce parallel jobs
    pre_dispatch='2*n_jobs',  # Limit memory usage
    error_score='raise'
)

# 2. Randomized Search - tries random combinations
RANDOM_SEARCH_ITERATIONS = 5
random_search = RandomizedSearchCV(
    pipeline,
    param_distributions,
    cv=cv,
    n_iter=RANDOM_SEARCH_ITERATIONS,
    verbose=1,
    return_train_score=True,
    n_jobs=4,   # Reduce parallel jobs
    pre_dispatch='2*n_jobs',  # Limit memory usage
    error_score='raise'  # Raise errors instead of crashing
)

# 3. Successive Halving Grid Search - eliminates poor performers early
halving_grid = HalvingGridSearchCV(
    pipeline,
    param_distributions,
    cv=cv,
    factor=3,  # reduction factor
    resource='n_samples',  # what to reduce
    min_resources='exhaust',  # min number of samples
    verbose=1,
    return_train_score=True,
    n_jobs=-1
)

# 4. Successive Halving Random Search - combines random search with successive halving
halving_random = HalvingRandomSearchCV(
    pipeline,
    param_distributions,
    cv=cv,
    n_candidates=20,  # number of parameter settings that are sampled
    factor=3,  # reduction factor
    resource='n_samples',  # what to reduce -> TRY OTHER RESOURCES -> N_ITERATIONS
    min_resources='exhaust',  # min number of samples
    verbose=1,
    return_train_score=True,
    n_jobs=-1
)

DIRECTORY = "res_pycatch"  # Directory to save results


search_strategy = random_search  # Choose the search strategy to use
SAVE_FILE = "randomized_search_" + str(RANDOM_SEARCH_ITERATIONS) + "_results.csv"  # File to save results
#search_strategy = grid_search  # Uncomment to use GridSearchCV
#SAVE_FILE = "grid_search_results.csv"  # File to save results
# search_strategy = halving_grid  # Uncomment to use HalvingGridSearchCV
# SAVE_FILE = "halving_grid_search_results.csv"  # File to save results
# search_strategy = halving_random  # Uncomment to use HalvingRandomSearchCV
# SAVE_FILE = "halving_random_search_results.csv"  # File to save results

# Measure execution time
print(f"\n--- Starting {search_strategy.__class__.__name__} fitting ---")
start_time = time.time()

# Fit with original data
search_strategy.fit(train_df, y_train)  # y_train is a dummy

# Calculate execution time
execution_time = time.time() - start_time
hours, remainder = divmod(execution_time, 3600)
minutes, seconds = divmod(remainder, 60)

print(f"\n--- {search_strategy.__class__.__name__} Results ---")
print(f"Execution time: {int(hours):02d}h {int(minutes):02d}m {seconds:.2f}s")


# Create a DataFrame with all results for better analysis
results_df = pd.DataFrame(search_strategy.cv_results_)

# Optionally, save full results to CSV for further analysis
results_df.to_csv(f"{DIRECTORY}/{SAVE_FILE}", index=False)
print(f"\nFull results saved to {SAVE_FILE}")

# Get best estimator and its feature importances
best_pipeline = search_strategy.best_estimator_
best_classifier = best_pipeline.named_steps['classifier']
feature_names = best_pipeline.named_steps['feature_selection'].get_feature_names()

# Get feature importances from the LightGBM model
importances = best_classifier.base_estimator.feature_importances_

# Create DataFrame with decoded feature names and importances
feature_importance_df = pd.DataFrame({
    'feature': feature_names[len(COLUMN_CONFIG['protected_cols']):],
    'importance': importances
})

# Sort by importance and display top features
feature_importance_df = feature_importance_df.sort_values('importance', ascending=False)

feature_importance_df.to_csv(f"{DIRECTORY}/importance_{SAVE_FILE}", index=False)
print(f"\nFeature_importance stored in importance_{SAVE_FILE}")

print("\n--- Best Model Feature Importances ---")
print("-" * 50)
print(feature_importance_df.head(10))
print("\n--- Best Parameters ---")
print("-" * 50)
print(search_strategy.best_params_)
print(f"\nBest score: {search_strategy.best_score_:.4f}")


best_results = {
    'best_params': search_strategy.best_params_,
    'best_score': float(search_strategy.best_score_)  # Convert numpy float to Python float for JSON serialization
}

json_filename = f"{DIRECTORY}/best_results_{SAVE_FILE.replace('.csv', '.json')}"
with open(json_filename, 'w') as f:
    json.dump(best_results, f, indent=4)

print(f"\nBest parameters and score saved to {json_filename}")












'''
# Get the best pipeline and transform the test data
best_pipeline = search_strategy.best_estimator_
test_transformed = test_df.copy()
for name, step in best_pipeline.named_steps.items():
    if name != 'classifier':
        test_transformed = step.transform(test_transformed)

# Calculate and print the final score
print(f"\nTest Accuracy: {best_pipeline.score(test_transformed):.4f}")


# Create a nicer display of the best parameters
print("\n--- Best Parameters ---")
for param, value in search_strategy.best_params_.items():
    print(f"{param}: {value}")

# Display the top 5 best parameter combinations
print("\n--- Top 5 Best Parameter Combinations ---")
top_params = results_df.sort_values('mean_test_score', ascending=False).head(5)
for i, row in top_params.iterrows():
    print(f"\nRank {i+1} - Score: {row['mean_test_score']:.4f}")
    for param, value in row['params'].items():
        print(f"  {param}: {value}")

# Display the worst 5 parameter combinations
print("\n--- 5 Worst Parameter Combinations (Potential Failures) ---")
worst_params = results_df.sort_values('mean_test_score').head(5)
for i, row in worst_params.iterrows():
    print(f"\nRank {len(results_df)-i} - Score: {row['mean_test_score']:.4f}")
    for param, value in row['params'].items():
        print(f"  {param}: {value}")

'''