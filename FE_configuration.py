from utils import train_test_split_by_time
from lightgbm import LGBMClassifier


# function to train a light gbm and return the model
def train_model(X_train, y_train):

    # transform the date column from datetime to other type
    X_train['DateTime'] = X_train['DateTime'].astype('int64')
    model = LGBMClassifier()
    model.fit(X_train, y_train)

    return model


def feature_extraction_preprocessing(data, COLUMN_CONFIG):
    train_df, _ = train_test_split_by_time(data, train_ratio=0.7)
    X_train, y_train = train_df.drop(columns=COLUMN_CONFIG['target_col']), train_df[COLUMN_CONFIG['target_col']]

    model = train_model(X_train, y_train)

    # get top 10 features from the model
    feature_importances = model.feature_importances_
    feature_names = X_train.columns
    top_features = sorted(zip(feature_importances, feature_names), reverse=True)[:10]
    top_feature_names = [feature[1] for feature in top_features]
    print("Top features:", top_feature_names)
    # use tsfel transformers to extract features : TODO