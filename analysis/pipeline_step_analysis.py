import pandas as pd
import os

# ...existing code...

# Create tables directory if it doesn't exist
os.makedirs('tables', exist_ok=True)

# Function to generate analysis and latex table for each parameter
def generate_parameter_analysis(results, param_name, param_columns):
    # Group by the first parameter (strategy/n_lags/etc)
    analysis = results.groupby(param_columns).agg({
        'mean_test_score': ['mean', 'std'],
        'mean_train_score': ['mean', 'std']
    }).round(4)
    
    # Rename columns for better readability
    analysis.columns = [
        'test_score_mean', 'test_score_std',
        'train_score_mean', 'train_score_std'
    ]
    
    if param_name == 'feature_extraction':
        # Extract just the framework name from the tuple string
        analysis.index = analysis.index.map(lambda x: (x[0] , x[1][2:].split('\'')[0]))
        analysis.index.names = ['n\_lags', 'extractor']
    elif param_name == 'feature_selection':
        analysis.index.names = ['strategy', 'threshold']
    elif param_name == 'imputation':
         analysis.index.name = 'strategy, params'
    
    #analysis.index.name = None
        
    # Generate LaTeX table
    latex_table = (
        "\\begin{table}\n"
        f"\\caption{{{param_name.replace('_', ' ').title()} Analysis}}\n"
        f"\\label{{tab:{param_name}_analysis}}\n"
        "\\begin{tabular}{lllll}\n"
        "\\toprule\n"
        f"({str(analysis.index.names)[1:-1]}) & Test Score (Mean) & Test Score (Std. Dev.) & Train Score (Mean) & Train Score (Std. Dev.) \\\\\n"
        "\\midrule\n"
    )
    
    # Add data rows
    for idx, row in analysis.iterrows():
        latex_table += f"\\textbf{{{idx}}} & {row['test_score_mean']:.4f} & {row['test_score_std']:.4f} & {row['train_score_mean']:.4f} & {row['train_score_std']:.4f} \\\\\n"
    
    # Add table footer
    latex_table += (
        "\\bottomrule\n"
        "\\end{tabular}\n"
        "\\end{table}"
    )
    
    # Save to file
    with open(f'tables/{param_name}_analysis.tex', 'w') as f:
        f.write(latex_table)
    
    print(f"\n{param_name.title()} Analysis:")
    print(analysis.index)

step_params ={ 
    'feature_extraction' : ['param_feature_extraction__n_lags','param_feature_extraction__params'],
    'feature_selection': ['param_feature_selection__strategy','param_feature_selection__threshold'],
    'imputation': ['param_imputation__params']
}
# Load the results
results = pd.read_csv('final_results.csv')
# Generate analysis for each parameter
for param_name, param_columns in step_params.items():
    generate_parameter_analysis(results, param_name, param_columns)