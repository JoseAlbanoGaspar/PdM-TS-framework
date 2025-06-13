import pandas as pd
from lightgbm import LGBMClassifier

# Load dataset
raw_df = pd.read_pickle("Datasets/final_dataset.pkl")

# Dataset column configuration
COLUMN_CONFIG = {
    'primary_key': ['ProcessId', 'DateTime'],
    'time_col': 'DateTime',
    'target_col': 'event',
    'protected_cols': ['ProcessId', 'DateTime', 'event'],
    'id_col': 'ProcessId',
}

# Feature extraction configuration
N_FEATURES = 10
N_TSFEL_FEATURES = 10
N_TSFRESH_FEATURES = 10

# Parameter grid from pipeline.py
param_distributions = {    
    # Imputation parameters
    'imputation__params': [
        ('interpolate', 'linear'),
        ('interpolate', ('polynomial', 2)),
        ('interpolate', ('spline', 2)),
        ('ffill', None),
    ],

    # Feature extraction parameters
    'feature_extraction__n_lags': [3, 5, 8],

    # Feature selection parameters
    'feature_selection__strategy': ['correlation', 'pca'],
    'feature_selection__threshold': [0.85, 0.9, 0.95],
    
    # Classifier parameters
    'classifier__base_estimator__max_depth': [3, 5, 7],
    'classifier__base_estimator__n_estimators': [25, 50, 100]
}

if __name__ == "__main__":
    from optimusPipe import optimize_pipeline
    
    # Configure the optimization
    base_classifier = LGBMClassifier()

    # Run optimization with random search
    results_random = optimize_pipeline(
        dataset=raw_df,
        column_config=COLUMN_CONFIG,
        n_features=N_FEATURES,
        n_tsfel_features=N_TSFEL_FEATURES,
        n_tsfresh_features=N_TSFRESH_FEATURES,
        base_classifier=base_classifier,
        search_strategy='random',
        n_jobs=2,
        random_iterations=2,
        output_dir='random_search_results',
        param_distributions=param_distributions,
        steps=['imputation', 'feature_extraction', 'feature_selection'],
        feature_extractors=['pycatch22']
    )