import os
import json
import pandas as pd
import shutil
from itertools import product
from sklearn.pipeline import Pipeline


from transformers import RegularityResampler, NAInterpolator, LagFeatureExtractor, CorrelationFeatureSelector, PCAFeatureSelector

def train_test_split_by_time(df, time_col='DateTime', id_col='ProcessId', train_ratio=0.7):
    """
    Splits the dataset into training and testing sets based on time.
    
    Parameters:
        df (pd.DataFrame): The input dataset.
        time_col (str): The name of the datetime column.
        id_col (str): The process identifier column.
        train_ratio (float): Proportion of oldest timestamps to use for training.
    
    Returns:
        train_df (pd.DataFrame): Training set (oldest 70% of timestamps per ProcessId).
        test_df (pd.DataFrame): Testing set (newest 30% of timestamps per ProcessId).
    """
    train_list = []
    test_list = []

    for process_id, group in df.groupby(id_col):
        group = group.sort_values(by=time_col)  # Ensure chronological order
        split_idx = int(len(group) * train_ratio)  # Determine the split index
        train_list.append(group.iloc[:split_idx])  # Oldest 70% for training
        test_list.append(group.iloc[split_idx:])  # Newest 30% for testing

    train_df = pd.concat(train_list).reset_index(drop=True)
    test_df = pd.concat(test_list).reset_index(drop=True)

    return train_df, test_df

########################################
#  Getting all pipelines combinations  #
########################################

# Define possible transformers
regularity_options = [None, RegularityResampler(freq='1H'), RegularityResampler(freq='2H')]
imputation_options = [NAInterpolator(method='linear')]#, NAForwardFill(), NABackwardFill()]
feature_extraction_options = [LagFeatureExtractor(n_lags=3), LagFeatureExtractor(n_lags=6)]
feature_selection_options = [CorrelationFeatureSelector(threshold=0.9), PCAFeatureSelector(variance_threshold=0.95)]

# Generate all valid pipeline configurations
pipeline_combinations = []
for reg, imp, feat_ext, feat_sel in product(regularity_options, imputation_options, feature_extraction_options, feature_selection_options):
    steps = []
    if reg: steps.append(('regularity', reg))
    steps.append(('imputation', imp))
    steps.append(('feature_extraction', feat_ext))
    steps.append(('feature_selection', feat_sel))

    pipeline_combinations.append(Pipeline(steps))



# Load raw dataset
raw_df = pd.read_pickle("Datasets/final_dataset.pkl")

# Split into train & test
train_df, test_df = train_test_split_by_time(raw_df, train_ratio=0.7)

# Create output directory if it doesn't exist
output_dir = "DatasetCleaned"
os.makedirs(output_dir, exist_ok=True)

# Clean the output directory before saving new datasets
if os.path.exists(output_dir):
    shutil.rmtree(output_dir)  # Delete everything inside the folder
os.makedirs(output_dir, exist_ok=True)  # Recreate empty folder


# Dictionary to store pipeline metadata
pipeline_metadata = {}

# Apply each pipeline and save results
for idx, pipeline in enumerate(pipeline_combinations):
    train_transformed = pipeline.fit_transform(train_df)
    test_transformed = pipeline.transform(test_df)

    # Save transformed datasets as pickle files
    train_path = os.path.join(output_dir, f'pipeline_{idx}_train.pkl')
    test_path = os.path.join(output_dir, f'pipeline_{idx}_test.pkl')
    train_transformed.to_pickle(train_path)
    test_transformed.to_pickle(test_path)

    # Store pipeline steps in metadata
    pipeline_metadata[idx] = str(pipeline.steps)
    print('Processed pipeline:', str(pipeline.steps))

# Save pipeline metadata to a JSON file
metadata_path = os.path.join(output_dir, "pipeline_metadata.json")
with open(metadata_path, "w") as f:
    json.dump(pipeline_metadata, f, indent=4)

print(f"✅ Processed {len(pipeline_combinations)} datasets and saved them in '{output_dir}'!")
print(f"📄 Pipeline metadata stored in '{metadata_path}'")
