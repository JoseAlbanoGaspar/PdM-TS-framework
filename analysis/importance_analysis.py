import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

tsfel = pd.read_csv('../res_tsfel/importance_grid_search_results.csv').head(10)
pycatch = pd.read_csv('../res_pycatch/importance_grid_search_results.csv').head(10)
tsfresh = pd.read_csv('../res_tsfresh/importance_grid_search_results.csv').head(10)

original = pd.read_pickle("../Datasets/final_dataset.pkl")

# for tsfel, pycatch, tsfresh generate a bar plot with number of features that are not on the original dataset
original_features = []
extracted_features = []
libraries = ['TSFEL', 'PyCatch22', 'TSFresh']

# Count features for each library
for df in [tsfel, pycatch, tsfresh]:
    orig = sum(df['feature'].isin(original.columns))
    extracted = len(df) - orig
    original_features.append(orig)
    extracted_features.append(extracted)

# Set up the plot
x = np.arange(len(libraries))
width = 0.35

fig, ax = plt.subplots(figsize=(10, 6))
rects1 = ax.bar(x - width/2, original_features, width, label='Original Features')
rects2 = ax.bar(x + width/2, extracted_features, width, label='Extracted Features')

# Customize the plot
ax.set_ylabel('Number of Features')
ax.set_title('Feature Distribution by Feature Extraction Framework')
ax.set_xticks(x)
ax.set_xticklabels(libraries)
ax.legend()

# Add value labels on top of bars
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{int(height)}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

autolabel(rects1)
autolabel(rects2)

plt.tight_layout()
plt.savefig('plots/feature_importance.png', dpi=300, bbox_inches='tight')
print('Feature importance plot saved as feature_importance.png')