from lightgbm import LGBMClassifier

from utils import  train_test_split_by_time
from .common import train_model, get_top_features

def get_top_dataset_features(data, COLUMN_CONFIG, n_features):
    """Extract top features from the original dataset using LightGBM."""
    # Split data and prepare for training
    train_df, _ = train_test_split_by_time(data, train_ratio=0.7)
    X_train, y_train = train_df.drop(columns=COLUMN_CONFIG['target_col']), train_df[COLUMN_CONFIG['target_col']]

    # Train model and get top features
    clf = LGBMClassifier()
    model = train_model(X_train, y_train, clf, COLUMN_CONFIG['protected_cols'])
    top_feature_names = get_top_features(model, X_train, n_features=n_features, protected_cols=COLUMN_CONFIG['protected_cols'])
    print(f"Top {n_features} features: {top_feature_names}")
    return top_feature_names, train_df

def get_subset_features_df(train_df, top_feature_names, COLUMN_CONFIG):
    """Extract subset of features from the original dataset based on top features."""
    subset_features_df = train_df[top_feature_names + COLUMN_CONFIG['protected_cols']]
    subset_features_df = subset_features_df.loc[:, ~subset_features_df.columns.duplicated()]

    return subset_features_df
