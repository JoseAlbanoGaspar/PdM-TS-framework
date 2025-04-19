from sklearn.pipeline import Pipeline
from tsfresh.feature_extraction import  EfficientFCParameters

from transformers.feature_extractors import TSFreshLagFeatureExtractor
from transformers.feature_selection import FeatureSelectionWrapper
from transformers.imputation import ImputationWrapper
from lightgbm import LGBMClassifier

from utils import clean_column_names, restore_column_names
from .common import train_model, get_top_features


def select_tsfresh_features(features, n_features=10):
    tsfresh_features = [s for s in features if '__' in s]

    tsfresh_feature_names = ['__'.join(s.split('__')[1:]) for s in tsfresh_features]

    tsfresh_features = []
    for feature_name in tsfresh_feature_names:
        if feature_name not in tsfresh_features:
            tsfresh_features.append(feature_name)
        if len(tsfresh_features) == n_features:
            break
    return tsfresh_features

def get_custumed_fc_parameters(final_tsfresh_features):
    fc_parameters = {}

    def convert_value(value):
        """Helper function to convert string values to appropriate types"""
        # Remove quotes if they exist
        if value.startswith('"') and value.endswith('"'):
            return value[1:-1]
        
        # Try converting to tuple if it contains parentheses
        if '(' in value and ')' in value:
            try:
                # Extract numbers from (2, 5, 10, 20) format
                nums = value.strip('()').split(',')
                return tuple(float(n.strip()) for n in nums)
            except:
                return value
                
        # Try converting to float/int
        try:
            # Convert to float first
            float_val = float(value)
            # If it's a whole number, convert to int
            if float_val.is_integer():
                return int(float_val)
            return float_val
        except ValueError:
            # Handle boolean values
            if value.lower() == 'true':
                return True
            if value.lower() == 'false':
                return False
            # If all else fails, return original string
            return value

    for feature_full_name in final_tsfresh_features:
        feature_name_parts = feature_full_name.split('__')
        feature_name = feature_name_parts[0]
        feature_attrs = feature_name_parts[1:]

        # If no attributes, set to None
        if not feature_attrs:
            if feature_name not in fc_parameters:
                fc_parameters[feature_name] = None
            continue

        # Create attributes dictionary
        attrs = {}
        for attr in feature_attrs:
            attr_parts = attr.split('_')
            attr_value = attr_parts[-1]
            attr_name = '_'.join(attr_parts[:-1])
            # Convert the attribute value to appropriate type
            attrs[attr_name] = convert_value(attr_value)

        # Handle multiple parameter combinations for same feature
        if feature_name in fc_parameters:
            if fc_parameters[feature_name] is None:
                fc_parameters[feature_name] = [attrs]
            elif isinstance(fc_parameters[feature_name], list):
                fc_parameters[feature_name].append(attrs)
            else:
                fc_parameters[feature_name] = [fc_parameters[feature_name], attrs]
        else:
            fc_parameters[feature_name] = [attrs]

    return fc_parameters



def extract_tsfresh_features(subset_features_df, COLUMN_CONFIG, n_tsfresh_features):
    """Extract TSFresh features from the top features."""
    subset_features_df = subset_features_df.copy()
    # Create and run TSFresh pipeline
    pipeline = Pipeline([
        ('imputation', ImputationWrapper(params=('interpolate', 'linear'), column_config=COLUMN_CONFIG)),
        ('feature_extraction', TSFreshLagFeatureExtractor(
            n_lags=4,
            default_fc_parameters=EfficientFCParameters(),
            column_config=COLUMN_CONFIG,
    )),
        ('feature_selection', FeatureSelectionWrapper(strategy='correlation', column_config=COLUMN_CONFIG))
    ])

    # Transform data and get top TSfresh features
    transformed_subset_df = pipeline.fit_transform(subset_features_df)
        
    clf = LGBMClassifier()

    transformed_subset_df, column_mapping = clean_column_names(transformed_subset_df)

    model = train_model(transformed_subset_df, transformed_subset_df[COLUMN_CONFIG['target_col']], 
                       clf, COLUMN_CONFIG['protected_cols'])
    
    tsfresh_feature_names = get_top_features(model=model, X_train=transformed_subset_df, protected_cols=COLUMN_CONFIG['protected_cols'])
    transformed_tsfresh_feature_names = restore_column_names(tsfresh_feature_names, column_mapping)
    final_tsfresh_features = select_tsfresh_features(transformed_tsfresh_feature_names, n_features=n_tsfresh_features)

    tsfresh_configuration = get_custumed_fc_parameters(final_tsfresh_features)

    return tsfresh_configuration
