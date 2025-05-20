import pandas as pd

# 1. Concatenating the grid search results
# Assuming you have three separate CSV files
df1 = pd.read_csv('../res_pycatch/grid_search_results.csv')
df2 = pd.read_csv('../res_tsfel/grid_search_results.csv')
df3 = pd.read_csv('../res_tsfresh/grid_search_results.csv')

# Concatenate all results
final_results = pd.concat([df1, df2, df3], ignore_index=True)

# Save concatenated results
final_results.to_csv('final_results.csv', index=False)