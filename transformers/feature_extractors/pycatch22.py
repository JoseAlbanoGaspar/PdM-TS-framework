from utils import DEFAULT_COLUMN_CONFIG
from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np

from sklearn.impute import SimpleImputer

import catch22_C



class PyCatch22LagFeatureExtractor(BaseEstimator, TransformerMixin):
    """
    Extracts features from lagged windows using pycatch22.
    """
    def __init__(self, n_lags=4, column_config=None, features=None, feature_names=None):
        self.n_lags = n_lags
        self.column_config = column_config or DEFAULT_COLUMN_CONFIG
        self.features = features # dataset features used for feature extraction
        self._selected_features = None
        self.feature_names = feature_names  # list of names

    def fit(self, X, y=None):
        # Pre-compute feature names using tsfresh
        if not self.feature_names:
            extracted_features = self._catch_n(X[self.column_config['id_col']].head(10), features=self.feature_names) # only to precompute feature names
            self.feature_names = list(extracted_features['names'])
        #print(f"Feature names: {self.feature_names}")
        return self
    
    def _process_group(self, group, numeric_cols):
        """Process a single group of data efficiently using vectorized operations"""
        result = group.copy()
        n_rows = len(group)
        
        if n_rows < self.n_lags:
            return pd.DataFrame()  # Return empty frame if not enough data
        
        # Create feature columns more efficiently
        feature_columns = {}
        
        for col in numeric_cols:
            values = group[col].values
            # Create sliding windows using stride tricks for better memory efficiency
            windows = np.lib.stride_tricks.sliding_window_view(
                values,
                window_shape=self.n_lags
            )

            #print(f"Windows shape: {windows.shape}")
            #print(f"Windows values: {windows}")
            # Pre-allocate arrays for features
            for feat_name in self.feature_names:
                feature_columns[f"{col}_{feat_name}"] = np.full(n_rows, np.nan)
            #print(f"Feature columns: {feature_columns}")
            #print("Windows for column", col, ":\n")
            for i in range(len(windows) - 1):
                # Extract features for each window
                window = windows[i]                
                try:
                    features = self._catch_n(window, features=self.feature_names)
                    features = {name: value for name, value in zip(features['names'], features['values'])}

                    print(f"Extracted features for window {i}:")
                    print("Window:\n", window)
                    print("Features:\n", features)
                    #print(features)

                    # Convert features to a dictionary for easier access

                    # Update feature columns with extracted features
                    for feat_name in self.feature_names:
                            feature_columns[f"{col}_{feat_name}"][i + self.n_lags] = features[f"{feat_name}"]
                except Exception as e:
                    print(f"Error processing window at index {i}: {e}")
                    print(f"Window shape: {window.shape}, Window values: {window}")
        
        # Create features DataFrame all at once
        features_df = pd.DataFrame(
            feature_columns,
            index=result.index
        )
        
        # Combine with original data and remove rows without features
        result = pd.concat([result, features_df], axis=1)
        return result.iloc[self.n_lags:]
    
    def transform(self, X):
        X = X.copy()
        primary_key = self.column_config['primary_key']
        time_col = self.column_config['time_col']
        protected_cols = self.column_config['protected_cols']
        
        # Get numeric columns or use specified features
        if not self.features:
            numeric_cols = [col for col in X.select_dtypes(include=[np.number]).columns 
                        if col not in protected_cols]
        else:
            numeric_cols = self.features
        
        # Get id columns
        id_cols = [col for col in primary_key if col != time_col]
        
        # Sort by time within each group
        X = X.sort_values(by=primary_key)
        
        if not id_cols:
            # Single time series case
            result = self._process_group(X, numeric_cols)
        else:
            # Multiple time series case
            groups = []
            for _, group in X.groupby(id_cols):
                processed_group = self._process_group(group, numeric_cols)
                groups.append(processed_group)
            result = pd.concat(groups, ignore_index=True)

        non_protected_cols = [col for col in result.columns if col not in protected_cols]
        # Drop columns that are all NA
        all_na_cols = result[non_protected_cols].columns[result[non_protected_cols].isna().all()].tolist()
        if all_na_cols:
            #print(f"Dropping columns with all NA values:\n{all_na_cols}")
            result = result.drop(columns=all_na_cols)
            non_protected_cols = [col for col in result.columns if col not in protected_cols]

        na_cols = result[non_protected_cols].columns[result[non_protected_cols].isna().any()].tolist()
        if na_cols:
            #print(f"Columns with NA values:\n{na_cols}")
            print("\nNA count per column:")
            print(result[na_cols].isna().sum())

        # Split data into protected and non-protected columns
        protected_data = result[protected_cols]
        non_protected_data = result.drop(columns=protected_cols)
        non_protected_data = non_protected_data.replace([np.inf, -np.inf], np.nan)

        # Apply SimpleImputer to non-protected columns
        imputer = SimpleImputer(strategy='mean')
        imputed_data = imputer.fit_transform(non_protected_data)

        # Convert back to DataFrame with original column names
        imputed_df = pd.DataFrame(
            imputed_data, 
            columns= non_protected_data.columns,
            #index=result.index
        )
        
        # Combine protected and imputed data
        final_result = pd.concat([protected_data, imputed_df], axis=1)
        
        #print("Feature names:\n", self.feature_names)
        #print("Columns:\n", final_result.columns)

        return final_result # Preserve original column order
    
    def _catch_n(self, data, catch24=False, features=None):
        '''
        Extract the catch22 feature set from an input time series.
        if list of features provided in features, only computes the features in the list.
        Otherwise, computes all features.

        Parameters
        ----------
        data : array_like
            Input time-series data.
        catch24 : bool, optional
            If True, include the two catch24 features (mean and standard deviation) in the output.
        short_names : bool, optional
            If True, also include the short names of the features in the output.

        '''
        if not features:
            features = [
                'DN_HistogramMode_5',
                'DN_HistogramMode_10',
                'CO_f1ecac',
                'CO_FirstMin_ac',
                'CO_HistogramAMI_even_2_5',
                'CO_trev_1_num',
                'MD_hrv_classic_pnn40',
                'SB_BinaryStats_mean_longstretch1',
                'SB_TransitionMatrix_3ac_sumdiagcov',
                'PD_PeriodicityWang_th0_01',
                'CO_Embed2_Dist_tau_d_expfit_meandiff',
                'IN_AutoMutualInfoStats_40_gaussian_fmmi',
                'FC_LocalSimple_mean1_tauresrat',
                'DN_OutlierInclude_p_001_mdrmd',
                'DN_OutlierInclude_n_001_mdrmd',
                'SP_Summaries_welch_rect_area_5_1',
                'SB_BinaryStats_diff_longstretch0',
                'SB_MotifThree_quantile_hh',
                'SC_FluctAnal_2_rsrangefit_50_1_logi_prop_r1',
                'SC_FluctAnal_2_dfa_50_1_2_logi_prop_r1',
                'SP_Summaries_welch_rect_centroid',
                'FC_LocalSimple_mean3_stderr'
            ]


        if catch24:
            features.append('DN_Mean')
            features.append('DN_Spread_Std')

        data = list(data)
        featureOut = []
        for f in features:
            featureFun = getattr(catch22_C, f)
            featureOut.append(featureFun(data))

      
        return {'names': features, 'values': featureOut}
