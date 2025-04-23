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

# Meta-modelo that receives the model and ignores a y_dummy
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

    # TODO AUC -> chamar o predict_proba para obter probabilidades e depois o auc_score -> DONE
    def score(self, X, y=None):
        y_true = X[self.actual_target_col]
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
    actual_target_col='event'
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
    'classifier__base_estimator__max_depth': [3, 5, 7, 9, 11],
    'classifier__base_estimator__n_estimators': [50, 100, 200, 300, 400, 500]
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
    n_jobs=-1
)

# 2. Randomized Search - tries random combinations
RANDOM_SEARCH_ITERATIONS = 200
random_search = RandomizedSearchCV(
    pipeline,
    param_distributions,
    cv=cv,
    n_iter=RANDOM_SEARCH_ITERATIONS,
    verbose=1,
    return_train_score=True,
    n_jobs=-1
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

search_strategy = random_search  # Choose the search strategy to use
SAVE_FILE = "randomized_search_" + str(RANDOM_SEARCH_ITERATIONS) + "_results.csv"  # File to save results
# search_strategy = grid_search  # Uncomment to use GridSearchCV
# SAVE_FILE = "grid_search_results.csv"  # File to save results
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

print("\n--- RandomizedSearchCV Results ---")
print(f"Execution time: {int(hours):02d}h {int(minutes):02d}m {seconds:.2f}s")


# Create a DataFrame with all results for better analysis
results_df = pd.DataFrame(search_strategy.cv_results_)

# Optionally, save full results to CSV for further analysis
results_df.to_csv(f"results/{SAVE_FILE}", index=False)
print(f"\nFull results saved to {SAVE_FILE}")


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

