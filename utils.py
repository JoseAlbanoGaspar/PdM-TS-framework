import pandas as pd


DEFAULT_COLUMN_CONFIG = {
    'primary_key': ['ProcessId', 'DateTime'],
    'time_col': 'DateTime',
    'target_col': 'event',
    'protected_cols': ['ProcessId', 'DateTime', 'event']
}


def train_test_split_by_time(df, time_col='DateTime', id_col='ProcessId', train_ratio=0.7):
    """
    Splits the dataset into training and testing sets based on time.
    """
    train_list, test_list = [], []

    for process_id, group in df.groupby(id_col):
        group = group.sort_values(by=time_col)
        split_idx = int(len(group) * train_ratio)
        train_list.append(group.iloc[:split_idx])
        test_list.append(group.iloc[split_idx:])

    train_df = pd.concat(train_list).reset_index(drop=True)
    test_df = pd.concat(test_list).reset_index(drop=True)

    return train_df, test_df