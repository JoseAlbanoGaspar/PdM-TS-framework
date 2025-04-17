import pandas as pd


DEFAULT_COLUMN_CONFIG = {
    'primary_key': ['ProcessId', 'DateTime'],
    'time_col': 'DateTime',
    'target_col': 'event',
    'protected_cols': ['ProcessId', 'DateTime', 'event'],
    'id_col': 'ProcessId',
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


def clean_column_names(df):
    """
    Converts all special characters in column names to hex encoding.
    
    Args:
        df: pandas DataFrame
    Returns:
        cleaned_df: DataFrame with cleaned column names
        column_mapping: dict mapping original to cleaned names
    """
    df = df.copy()
    
    def encode_special_chars(name):
        # Convert all non-alphanumeric characters (except underscore) to hex representation
        return ''.join(c if c.isalnum() or c == '_' else f'_x{ord(c):02x}_' for c in name)
    
    column_mapping = {col: encode_special_chars(col) for col in df.columns}
    df.columns = [column_mapping[col] for col in df.columns]
    return df, column_mapping

def restore_column_names(column_names, column_mapping):
    """
    Restores original column names using mapping.
    
    Args:
        column_names: list of column names to restore
        column_mapping: dict mapping cleaned to original names
    Returns:
        restored_names: list of restored original column names
    """
    reverse_mapping = {v: k for k, v in column_mapping.items()}
    restored_names = [reverse_mapping.get(col, col) for col in column_names]
    return restored_names
