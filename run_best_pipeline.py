import os
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import precision_score, recall_score, confusion_matrix, roc_auc_score, accuracy_score
from FE_configuration import feature_extraction_preprocessing
from transformers import ImputationWrapper, FeatureExtractorWrapper, FeatureSelectionWrapper
from utils import train_test_split_by_time
from metamodel import MetaClassifier

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

# Feature extraction preprocessing (use a small subset for speed)
sub_train_df, _ = train_test_split_by_time(raw_df, train_ratio=0.1)
tsfel_config_file, top_features, tsfresh_fc_parameters = feature_extraction_preprocessing(
    sub_train_df, COLUMN_CONFIG, N_FEATURES, N_TSFEL_FEATURES, N_TSFRESH_FEATURES
)

# Best parameters from search
best_params = {
    "classifier__base_estimator__max_depth": 5,
    "classifier__base_estimator__n_estimators": 100,
    "feature_extraction__n_lags": 5,
    "feature_extraction__params": ('pycatch22', {}),
    "feature_selection__strategy": "correlation",
    "feature_selection__threshold": 0.85,
    "imputation__params": ('interpolate', 'linear')
}

# --- BEST PIPELINE ---
meta_clf = MetaClassifier(
    base_estimator=LGBMClassifier(
        max_depth=best_params["classifier__base_estimator__max_depth"],
        n_estimators=best_params["classifier__base_estimator__n_estimators"]
    ),
    column_config=COLUMN_CONFIG
)

best_pipeline = Pipeline([
    ('imputation', ImputationWrapper(params=best_params["imputation__params"], column_config=COLUMN_CONFIG)),
    ('feature_extraction', FeatureExtractorWrapper(
        params=best_params["feature_extraction__params"],
        n_lags=best_params["feature_extraction__n_lags"],
        column_config=COLUMN_CONFIG,
        features=top_features
    )),
    ('feature_selection', FeatureSelectionWrapper(
        strategy=best_params["feature_selection__strategy"],
        threshold=best_params["feature_selection__threshold"],
        column_config=COLUMN_CONFIG
    )),
])

# Fit and transform train data
X_train_best = best_pipeline.fit_transform(train_df)
X_test_best = best_pipeline.transform(test_df)
y_test = X_test_best[COLUMN_CONFIG['target_col']]

# For MetaClassifier, X must include protected columns and target_col for encoding
meta_clf.fit(X_train_best)
y_pred_best = meta_clf.predict(X_test_best)
y_proba_best = meta_clf.predict_proba(X_test_best)[:, 1]

precision_best = precision_score(y_test, y_pred_best)
recall_best = recall_score(y_test, y_pred_best)
accuracy_best = accuracy_score(y_test, y_pred_best)
auc_best = roc_auc_score(y_test, y_proba_best)
tn, fp, fn, tp = confusion_matrix(y_test, y_pred_best).ravel()
fp_best, fn_best = fp, fn

# --- BASELINE PIPELINE (no feature extraction) ---
baseline_pipeline = Pipeline([
    ('imputation', ImputationWrapper(params=best_params["imputation__params"], column_config=COLUMN_CONFIG)),
    ('feature_selection', FeatureSelectionWrapper(
        strategy=best_params["feature_selection__strategy"],
        threshold=best_params["feature_selection__threshold"],
        column_config=COLUMN_CONFIG
    )),
])

# Fit and transform train data
X_train_baseline = baseline_pipeline.fit_transform(train_df)
X_test_baseline = baseline_pipeline.transform(test_df)
y_test = X_test_baseline[COLUMN_CONFIG['target_col']]

# Train MetaClassifier on baseline features
meta_clf_baseline = MetaClassifier(
    base_estimator=LGBMClassifier(
        max_depth=best_params["classifier__base_estimator__max_depth"],
        n_estimators=best_params["classifier__base_estimator__n_estimators"]
    ),
    column_config=COLUMN_CONFIG
)
meta_clf_baseline.fit(X_train_baseline)
y_pred_baseline = meta_clf_baseline.predict(X_test_baseline)
y_proba_baseline = meta_clf_baseline.predict_proba(X_test_baseline)[:, 1]

precision_baseline = precision_score(y_test, y_pred_baseline)
recall_baseline = recall_score(y_test, y_pred_baseline)
accuracy_baseline = accuracy_score(y_test, y_pred_baseline)
auc_baseline = roc_auc_score(y_test, y_proba_baseline)
tn, fp, fn, tp = confusion_matrix(y_test, y_pred_baseline).ravel()
fp_baseline, fn_baseline = fp, fn

# --- PRINT RESULTS AS DATAFRAME ---
results_df = pd.DataFrame({
    "Precision": [precision_best, precision_baseline],
    "Recall": [recall_best, recall_baseline],
    "Accuracy": [accuracy_best, accuracy_baseline],
    "AUC": [auc_best, auc_baseline],
    "False Positives": [fp_best, fp_baseline],
    "False Negatives": [fn_best, fn_baseline]
}, index=["Best", "Baseline"])

print("\nComparison of Best and Baseline Pipelines:\n")
print(results_df)

# --- LATEX TABLE ---
table = r"""
\begin{table}[ht]
\centering
\begin{tabular}{lccccc}
\hline
Model & Precision & Recall & Accuracy & AUC & False Positives & False Negatives \\
\hline
Best & %.4f & %.4f & %.4f & %.4f & %d & %d \\
Baseline & %.4f & %.4f & %.4f & %.4f & %d & %d \\
\hline
\end{tabular}
\caption{Comparison of best pipeline and baseline pipeline on test set.}
\label{tab:best_baseline_comparison}
\end{table}
""" % (
    precision_best, recall_best, accuracy_best, auc_best, fp_best, fn_best,
    precision_baseline, recall_baseline, accuracy_baseline, auc_baseline, fp_baseline, fn_baseline
)

os.makedirs("analysis/tables", exist_ok=True)
with open("analysis/tables/best_baseline_comparison.tex", "w") as f:
    f.write(table)

print("\nLatex table saved to analysis/tables/best_baseline_comparison.tex")