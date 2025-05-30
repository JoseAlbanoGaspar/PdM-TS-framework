import sys
import pandas as pd
import pickle

def check_nulls_in_pickle(pkl_path):
    # Try to load the pickle file
    try:
        with open(pkl_path, 'rb') as f:
            obj = pickle.load(f)
    except Exception as e:
        print(f"Error loading pickle file: {e}")
        return

    # Try to convert to DataFrame if not already
    if isinstance(obj, pd.DataFrame):
        df = obj
    elif isinstance(obj, dict) or isinstance(obj, list):
        try:
            df = pd.DataFrame(obj)
        except Exception as e:
            print(f"Cannot convert object to DataFrame: {e}")
            return
    else:
        print("Loaded object is not a DataFrame, dict, or list.")
        return

    # Print the number of null values per column
    null_counts = df.isnull().sum()
    print("Null values per column:")
    print(null_counts)

    # Optionally, print if any nulls exist at all
    if null_counts.any():
        print("The file contains null values.")
    else:
        print("The file does NOT contain null values.")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python test_null_values.py <path_to_pkl_file>")
    else:
        check_nulls_in_pickle(sys.argv[1])