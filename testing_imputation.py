import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from transformers import ImputationWrapper

# Configuration
COLUMN_CONFIG = {
    'primary_key': ['ProcessId', 'DateTime'],
    'time_col': 'DateTime',
    'target_col': 'event',
    'protected_cols': ['ProcessId', 'DateTime', 'event']
}

# List of imputation parameters to test
imputation_params = [
    ('interpolate', 'linear'),
    ('interpolate', ('polynomial', 2)),
    ('interpolate', ('polynomial', 3)),
    #('interpolate', ('spline', 2)),
    #('interpolate', ('spline', 3)),
    ('ffill', None),
]

def test_imputation(df, params):
    """Test a single imputation method"""
    pipeline = Pipeline([
        ('imputation', ImputationWrapper(
            params=params, 
            column_config=COLUMN_CONFIG
        ))
    ])
    
    # Apply transformation
    try:
        result = pipeline.fit_transform(df)
        
        # Check for null values
        null_counts = result.isnull().sum()
        has_nulls = null_counts.sum() > 0
        
        print(f"\nTesting imputation method: {params}")
        print("-" * 50)
        print(f"Input shape: {df.shape}")
        print(f"Output shape: {result.shape}")
        print(f"Has null values: {has_nulls}")
        
        if has_nulls:
            print("\nColumns with null values:")
            for col, count in null_counts[null_counts > 0].items():
                print(f"- {col}: {count} nulls")
        
        return result, has_nulls
        
    except Exception as e:
        print(f"\nError testing {params}:")
        print(f"Error message: {str(e)}")
        return None, True

def main():
    # Load dataset
    print("Loading dataset...")
    raw_df = pd.read_pickle("Datasets/final_dataset.pkl")
    
    # Print initial null values
    print("\nInitial dataset stats:")
    print("-" * 50)
    print(f"Shape: {raw_df.shape}")
    print("\nColumns with null values:")
    null_counts = raw_df.isnull().sum()
    for col, count in null_counts[null_counts > 0].items():
        print(f"- {col}: {count} nulls")
    
    # Test each imputation method
    results = {}
    for params in imputation_params:
        result_df, has_nulls = test_imputation(raw_df.copy(), params)
        results[str(params)] = {
            'success': result_df is not None,
            'has_nulls': has_nulls
        }
    
    # Print summary
    print("\nSummary:")
    print("-" * 50)
    for method, result in results.items():
        status = "✓" if result['success'] and not result['has_nulls'] else "✗"
        print(f"{status} {method}")

if __name__ == "__main__":
    main()