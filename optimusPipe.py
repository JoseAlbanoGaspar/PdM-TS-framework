from lightgbm import LGBMClassifier
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.model_selection import (
    RandomizedSearchCV,
    GridSearchCV,
    TimeSeriesSplit
)
import time

from FE_configuration import feature_extraction_preprocessing
from transformers import (
    ImputationWrapper,
    FeatureExtractorWrapper,
    FeatureSelectionWrapper,
    RegularityResampler,
    NormalizationWrapper
)

from metamodel import MetaClassifier

from utils import train_test_split_by_time
import json
import os

def optimize_pipeline(
    dataset,
    column_config,
    n_features=10,
    n_tsfel_features=10, 
    n_tsfresh_features=10,
    base_classifier=LGBMClassifier(),
    search_strategy='grid',
    n_jobs=5,
    random_iterations=100,
    output_dir='results',
    steps=['regularity','imputation', 'feature_extraction', 'feature_selection', 'normalization'],  # Default steps to include in the pipeline,
    param_distributions=None,
    feature_extractors=['tsfel']
    ):
    """
    Optimizes an ML pipeline using the specified search strategy.
    
    Args:
        dataset (pd.DataFrame): Input dataset
        column_config (dict): Column configuration dictionary
        n_features (int): Number of features to extract
        n_tsfel_features (int): Number of TSFEL features
        n_tsfresh_features (int): Number of TSFresh features 
        base_classifier: Base classifier to use
        search_strategy (str): One of 'grid', 'random', 'halving_grid', 'halving_random'
        n_jobs (int): Number of parallel jobs
        random_iterations (int): Number of iterations for random search
        output_dir (str): Directory to save results
        steps (list): List of preprocessing steps to include in the pipeline
    Returns:
        dict: Results including best parameters, scores and feature importances
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Split dataset into train & test
    train_df, test_df = train_test_split_by_time(dataset, train_ratio=0.7)
    X_train, y_train = train_df.drop(columns=['event']), train_df['event']
    
    # Feature extraction preprocessing
    sub_train_df, _ = train_test_split_by_time(dataset, train_ratio=0.1)
    tsfel_config, top_features, tsfresh_fc_parameters = feature_extraction_preprocessing(
        sub_train_df, column_config, n_features, n_tsfel_features, n_tsfresh_features
    )

    # Create pipeline
    meta_clf = MetaClassifier(base_estimator=base_classifier, column_config=column_config)
    pipeline_steps = []

    if 'regularity' in steps:
        pipeline_steps.append(('regularity', RegularityResampler(freq='1H')))
    if 'imputation' in steps:
        pipeline_steps.append(('imputation', ImputationWrapper(params=('interpolate', 'linear'), column_config=column_config))) 
    if 'feature_extraction' in steps:
        pipeline_steps.append(('feature_extraction', FeatureExtractorWrapper(
            params=('tsfel', {'config_file': tsfel_config}),
            n_lags=8,
            column_config=column_config,
            features=top_features
        )))

        feature_extraction_params = []
        if 'tsfel' in feature_extractors:
            feature_extraction_params.append(('tsfel', {'config_file': tsfel_config}))
        if 'tsfresh' in feature_extractors:
            feature_extraction_params.append(('tsfresh', {'default_fc_parameters': tsfresh_fc_parameters}))
        if 'pycatch22' in feature_extractors:
            feature_extraction_params.append(('pycatch22', {}))
        param_distributions['feature_extraction__params'] = feature_extraction_params

    if 'feature_selection' in steps:
        pipeline_steps.append(('feature_selection', FeatureSelectionWrapper(strategy='correlation', column_config=column_config)))
    if 'normalization' in steps:
        pipeline_steps.append(('normalization',NormalizationWrapper(scaler_method='minmax',column_config=column_config)))
    
    pipeline_steps.append(('classifier', meta_clf)) 
    
    pipeline = Pipeline(steps=pipeline_steps)

    # Configure cross-validation
    cv = TimeSeriesSplit(n_splits=5)

    # Select search strategy
    search_strategies = {
        'grid': 
            (GridSearchCV(pipeline, param_distributions, cv=cv, n_jobs=n_jobs),
             "grid_search_results.csv"),
        'random': (RandomizedSearchCV(pipeline, param_distributions, cv=cv, n_iter=random_iterations, n_jobs=n_jobs), 
                   "randomized_search_" + str(random_iterations) + "_results.csv"),
        }
    
    search_strategy, SAVE_FILE = search_strategies[search_strategy]

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
    results_df.to_csv(f"{output_dir}/{SAVE_FILE}", index=False)
    print(f"\nFull results saved to {SAVE_FILE}")

    # Get best estimator and its feature importances
    best_pipeline = search_strategy.best_estimator_
    best_classifier = best_pipeline.named_steps['classifier']
    feature_names = best_pipeline.named_steps['feature_selection'].get_feature_names()

    # Get feature importances from the LightGBM model
    importances = best_classifier.base_estimator.feature_importances_

    # Create DataFrame with decoded feature names and importances
    feature_importance_df = pd.DataFrame({
        'feature': feature_names[len(column_config['protected_cols']):],
        'importance': importances
    })

    # Sort by importance and display top features
    feature_importance_df = feature_importance_df.sort_values('importance', ascending=False)

    feature_importance_df.to_csv(f"{output_dir}/importance_{SAVE_FILE}", index=False)
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

    json_filename = f"{output_dir}/best_results_{SAVE_FILE.replace('.csv', '.json')}"
    with open(json_filename, 'w') as f:
        json.dump(best_results, f, indent=4)

    print(f"\nBest parameters and score saved to {json_filename}")