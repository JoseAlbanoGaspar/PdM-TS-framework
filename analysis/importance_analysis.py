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
plt.title('Feature Importance Comparison')
plt.xlabel('Feature Extraction Frameworks')
plt.ylabel('Number of Features')
plt.savefig('plots/feature_importance.png', dpi=300, bbox_inches='tight')
print('Feature importance plot saved as feature_importance.png')

extractors = [tsfel, pycatch, tsfresh]
extractor_names = ['TSFEL', 'PyCatch22', 'TSFresh']

# 3. Table Plot with Color-coded Labels
# --- Triangle plot for TSFEL and TSFresh (2 on top, 1 on bottom) ---
import matplotlib.gridspec as gridspec

# Prepare data for each extractor: TSFEL (top left), TSFresh (top right), PyCatch22 (bottom)
tables_data = []
for df in [tsfel, tsfresh, pycatch]:
    feature_names = df['feature'].tolist()
    importances = df['importance'].tolist()
    is_original = [f in original.columns for f in feature_names]
    labels = ['Original' if orig else 'Extracted' for orig in is_original]
    colors = ['#1f77b4' if orig else '#ff7f0e' for orig in is_original]
    # Put importance in the middle
    cell_text = [[feat, imp, label] for feat, imp, label in zip(feature_names, importances, labels)]
    cell_colors = [['white', 'white', color] for color in colors]
    tables_data.append((cell_text, cell_colors))

fig = plt.figure(figsize=(14, 9))
gs = gridspec.GridSpec(2, 4, width_ratios=[1, 3, 3, 1], height_ratios=[1, 1])

axes = []
axes.append(fig.add_subplot(gs[0, 1]))  # Top far left (TSFEL)
axes.append(fig.add_subplot(gs[0, 2]))  # Top far right (TSFresh)
axes.append(fig.add_subplot(gs[1, 1:3]))  # Bottom center (PyCatch22, spans both center columns)

for ax, (cell_text, cell_colors), name in zip(
    axes, tables_data, ['TSFEL', 'TSFresh', 'PyCatch22']
):
    ax.axis('off')
    table = ax.table(
        cellText=cell_text,
        cellColours=cell_colors,
        colLabels=['Feature', 'Importance', 'Type'],
        loc='center',
        cellLoc='center',
        colLoc='center'
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)
    table.auto_set_column_width([0, 1, 2])
    for (row, col), cell in table.get_celld().items():
        if col == 0:
            cell.set_width(0.65)
        elif col == 1:
            cell.set_width(0.18)
        elif col == 2:
            cell.set_width(0.17)
    ax.set_title(f'Top 10 Features: {name}', fontsize=12, pad=8)

plt.suptitle('Top 10 Features for each feature extractor', fontsize=15, y=0.98)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig('plots/feature_importance_names.png', dpi=300, bbox_inches='tight')
print('Feature importance names plot saved as feature_importance_names.png')