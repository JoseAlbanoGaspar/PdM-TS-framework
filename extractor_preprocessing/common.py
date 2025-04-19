def train_model(X_train, y_train, model, protected_cols=None):
    X_train = X_train.copy()
    
    X_train['DateTime'] = X_train['DateTime'].astype('int64')
    
    X_train.drop(columns=protected_cols, inplace=True, errors='ignore')

    model.fit(X_train, y_train)
    return model


def get_top_features(model, X_train, n_features=None, protected_cols=None):
    feature_importances = model.feature_importances_
    
    feature_names = [col for col in X_train.columns if col not in protected_cols]

    # If n_features is None, use all features
    n_features = len(feature_names) if n_features is None else n_features
    top_features = sorted(zip(feature_importances, feature_names), reverse=True)[:n_features]
    top_feature_names = [feature[1] for feature in top_features]
    return top_feature_names
