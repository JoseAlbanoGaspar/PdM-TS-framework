from sklearn.base import BaseEstimator, TransformerMixin
from transformers.feature_extractors.tsfel import TSFELLagFeatureExtractor
from transformers.feature_extractors.tsfresh import TSFreshLagFeatureExtractor

class FeatureExtractorWrapper(BaseEstimator, TransformerMixin):
    def __init__(self, params=None, n_lags=8, column_config=None, features=None):
        self.params = params
        self.n_lags = n_lags
        self.column_config = column_config
        self.features = features
        self.extractor = None

    def fit(self, X, y=None):
        strategy = self.params[0]  # Get strategy from first element of tuple
        params = self.params[1]    # Get parameters from second element of tuple
        
        if strategy == 'tsfel':
            self.extractor = TSFELLagFeatureExtractor(
                n_lags=self.n_lags,
                config_file=params['config_file'],
                features=self.features,
                column_config=self.column_config
            )
        elif strategy == 'tsfresh':
            self.extractor = TSFreshLagFeatureExtractor(
                n_lags=self.n_lags,
                default_fc_parameters=params['default_fc_parameters'],
                features=self.features,
                column_config=self.column_config
            )
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
        
        return self.extractor.fit(X, y)

    def transform(self, X):
        return self.extractor.transform(X)