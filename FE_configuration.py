from sklearn.pipeline import Pipeline
from transformers.feature_extraction import TSFELLagFeatureExtractor
from transformers.feature_selection import FeatureSelectionWrapper
from transformers.imputation import ImputationWrapper
from utils import train_test_split_by_time
from lightgbm import LGBMClassifier
import numpy as np

def train_model(X_train, y_train, model, protected_cols=None):
    X_train = X_train.copy()
    
    X_train['DateTime'] = X_train['DateTime'].astype('int64')
    
    X_train.drop(columns=protected_cols, inplace=True, errors='ignore')
  
    model.fit(X_train, y_train)
    return model


def get_top_features(model, X_train, n_features=None):
    feature_importances = model.feature_importances_
    feature_names = X_train.columns

    # If n_features is None, use all features
    n_features = len(feature_names) if n_features is None else n_features
    top_features = sorted(zip(feature_importances, feature_names), reverse=True)[:n_features]
    top_feature_names = [feature[1] for feature in top_features]
    return top_feature_names

def feature_extraction_preprocessing(data, COLUMN_CONFIG):
    train_df, _ = train_test_split_by_time(data, train_ratio=0.7)
    X_train, y_train = train_df.drop(columns=COLUMN_CONFIG['target_col']), train_df[COLUMN_CONFIG['target_col']]

    clf = LGBMClassifier()

    model = train_model(X_train, y_train, clf, COLUMN_CONFIG['protected_cols'])

    top_feature_names = get_top_features(model, X_train, n_features=10)
    subset_features_df = train_df[top_feature_names + COLUMN_CONFIG['protected_cols']]

    print("Top features:", top_feature_names)


    # use tsfel transformers to extract features
    pipeline = Pipeline([
    ('imputation', ImputationWrapper(params=('interpolate', 'linear'), column_config=COLUMN_CONFIG)),
    ('feature_extraction', TSFELLagFeatureExtractor(
        n_lags=4,
        domains=['statistical', 'temporal'], # all available domains 
        column_config=COLUMN_CONFIG
    )),
    ('feature_selection', FeatureSelectionWrapper(strategy='correlation', column_config=COLUMN_CONFIG))
    ])

    # remove duplicates from the subset_features_df
    subset_features_df = subset_features_df.loc[:, ~subset_features_df.columns.duplicated()]
    transformed_subset_df = pipeline.fit_transform(subset_features_df)

    model = train_model(transformed_subset_df, transformed_subset_df[COLUMN_CONFIG['target_col']], clf,  COLUMN_CONFIG['protected_cols'])
    tsfel_feature_names = get_top_features(model, transformed_subset_df) # returns all names

    print("TSFEL features:", tsfel_feature_names)
