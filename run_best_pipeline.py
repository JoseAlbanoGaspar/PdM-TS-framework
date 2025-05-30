import os
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import precision_score, recall_score, confusion_matrix, roc_auc_score, accuracy_score
from sklearn.tree import DecisionTreeClassifier
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
    "classifier__base_estimator__n_estimators": 50,
    "feature_extraction__n_lags": 3,
    "feature_extraction__params": ('pycatch22', {}),
    "feature_selection__strategy": "correlation",
    "feature_selection__threshold": 0.95,
    "imputation__params": ('interpolate', 'linear')
}

# --- BEST PIPELINE ---
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

tree_classifiers = {
    "LightGBM": LGBMClassifier(
        max_depth=best_params["classifier__base_estimator__max_depth"],
        n_estimators=best_params["classifier__base_estimator__n_estimators"]
    ),
    "RandomForest": RandomForestClassifier(
        max_depth=best_params["classifier__base_estimator__max_depth"],
        n_estimators=best_params["classifier__base_estimator__n_estimators"],
    ),
    "DecisionTree": DecisionTreeClassifier(
        max_depth=best_params["classifier__base_estimator__max_depth"],
        random_state=42
    ),
}

results = []

for clf_name, clf in tree_classifiers.items():
    meta_clf = MetaClassifier(
        base_estimator=clf,
        column_config=COLUMN_CONFIG
    )
    meta_clf.fit(X_train_best)
    y_pred = meta_clf.predict(X_test_best)
    y_proba = meta_clf.predict_proba(X_test_best)[:, 1]

    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    results.append({
        "Model": f"Best ({clf_name})",
        "Precision": precision,
        "Recall": recall,
        "Accuracy": accuracy,
        "AUC": auc,
        "False Positives": fp,
        "False Negatives": fn
    })


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
results.append({
    "Model": "Baseline (LightGBM)",
    "Precision": precision_baseline,
    "Recall": recall_baseline,
    "Accuracy": accuracy_baseline,
    "AUC": auc_baseline,
    "False Positives": fp,
    "False Negatives": fn
})

# --- PRINT RESULTS AS DATAFRAME ---
results_df = pd.DataFrame(results)
results_df.set_index("Model", inplace=True)
print("\nComparison of Best (multiple trees) and Baseline Pipelines:\n")
print(results_df)

# --- LATEX TABLE ---
table_header = r"""
\begin{table}[ht]
\centering
\begin{tabular}{lccccc}
\hline
Model & Precision & Recall & Accuracy & AUC & False Positives & False Negatives \\
\hline
"""
table_rows = ""
for idx, row in results_df.iterrows():
    table_rows += f"{idx} & {row['Precision']:.4f} & {row['Recall']:.4f} & {row['Accuracy']:.4f} & {row['AUC']:.4f} & {int(row['False Positives'])} & {int(row['False Negatives'])} \\\\\n"
table_footer = r"""\hline
\end{tabular}
\caption{Comparison of best pipeline (with multiple tree-based classifiers) and baseline pipeline on test set.}
\label{tab:best_baseline_comparison}
\end{table}
"""

latex_table = table_header + table_rows + table_footer

os.makedirs("analysis/tables", exist_ok=True)
with open("analysis/tables/best_baseline_comparison.tex", "w") as f:
    f.write(latex_table)

print("\nLatex table saved to analysis/tables/best_baseline_comparison.tex")