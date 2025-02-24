import os
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

# File paths (update these paths as needed)
csv_no_stability = '/cluster/project/math/akmete/MSc/Stabilized_Regression/In_Site/NoStability/LR/results.csv'
csv_stability = '/cluster/project/math/akmete/MSc/Stabilized_Regression/In_Site/WithStability/LR/results.csv'

# Load the two CSV files into DataFrames
df_no_stability = pd.read_csv(csv_no_stability)
df_stability = pd.read_csv(csv_stability)

# Merge the two DataFrames on the "site" column.
# Suffixes indicate which result came from which CSV.
df_merged = pd.merge(df_no_stability, df_stability, on='site', suffixes=('_no_stability', '_stability'))

# List of ensemble metrics to compare
ensemble_metrics = [
    'ensemble_mse_scaled', 
    'ensemble_rmse_scaled', 
    'ensemble_r2', 
    'ensemble_relative_error', 
    'ensemble_mae'
]

# Calculate the difference for each ensemble metric: (stability - no_stability)
for metric in ensemble_metrics:
    diff_metric = metric + '_diff'
    df_merged[diff_metric] = df_merged[f'{metric}_stability'] - df_merged[f'{metric}_no_stability']

# Save merged ensemble results (with differences) to a CSV file for reference
df_merged.to_csv('merged_ensemble_results.csv', index=False)
print("Merged ensemble results saved to merged_ensemble_results.csv")

# Create a directory to store ensemble plots
os.makedirs('plots_LR', exist_ok=True)

# Create scatter plots to compare each ensemble metric and save them separately
for metric in ensemble_metrics:
    plt.figure(figsize=(8, 6))
    sns.scatterplot(
        data=df_merged,
        x=f'{metric}_no_stability',
        y=f'{metric}_stability'
    )
    # Plot a diagonal line for reference
    min_val = min(df_merged[f'{metric}_no_stability'].min(), df_merged[f'{metric}_stability'].min())
    max_val = max(df_merged[f'{metric}_no_stability'].max(), df_merged[f'{metric}_stability'].max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='y=x')
    plt.title(f'Ensemble {metric} Comparison')
    plt.xlabel(f'No Stability {metric}')
    plt.ylabel(f'Stability {metric}')
    plt.legend()
    plt.tight_layout()
    # Save each plot with a unique file name
    plot_filename = os.path.join('plots_LR', f'{metric}_comparison.png')
    plt.savefig(plot_filename)
    plt.close()
    print(f"Plot for {metric} saved as {plot_filename}")


