import os
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

# File paths (update these paths as needed)
csv_no_stability = '/cluster/project/math/akmete/MSc/Stabilized_Regression/In_Site/NoStability/XGBoost/results.csv'
csv_stability = '/cluster/project/math/akmete/MSc/Stabilized_Regression/In_Site/WithStability/XGBoost/results.csv'

# Load the CSV files into DataFrames
df_no_stability = pd.read_csv(csv_no_stability)
df_stability    = pd.read_csv(csv_stability)

# Merge the DataFrames on the "site" column.
df_merged = pd.merge(df_no_stability, df_stability, on='site', suffixes=('_no_stability', '_stability'))

# List of ensemble metrics we want to compare
ensemble_metrics = ['ensemble_mse_scaled', 'ensemble_rmse']

# Calculate the difference for each ensemble metric (stability - no_stability)
for metric in ensemble_metrics:
    diff_metric = metric + '_diff'
    df_merged[diff_metric] = df_merged[f'{metric}_stability'] - df_merged[f'{metric}_no_stability']

# Save the merged results (with differences) to a CSV file
merged_output_path = 'merged_ensemble_results.csv'
df_merged.to_csv(merged_output_path, index=False)
print(f"Merged ensemble results saved to {merged_output_path}")

# Save summary statistics for the differences to a CSV file
diff_columns = [m + '_diff' for m in ensemble_metrics]
diff_stats = df_merged[diff_columns].describe()
stats_output_path = 'ensemble_diff_summary_stats.csv'
diff_stats.to_csv(stats_output_path)
print(f"Summary statistics for ensemble metric differences saved to {stats_output_path}")

# Create a directory to store plots
plots_dir = 'plots_XGB'
os.makedirs(plots_dir, exist_ok=True)

# Create scatter plots for each ensemble metric and save them
for metric in ensemble_metrics:
    plt.figure(figsize=(8, 6))
    sns.scatterplot(
        data=df_merged,
        x=f'{metric}_no_stability',
        y=f'{metric}_stability'
    )
    # Compute min and max values for the axes
    min_val = min(df_merged[f'{metric}_no_stability'].min(), df_merged[f'{metric}_stability'].min())
    max_val = max(df_merged[f'{metric}_no_stability'].max(), df_merged[f'{metric}_stability'].max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='y=x')
    plt.title(f'Ensemble {metric} Comparison')
    plt.xlabel(f'No Stability {metric}')
    plt.ylabel(f'Stability {metric}')
    plt.legend()
    plt.tight_layout()
    plot_path = os.path.join(plots_dir, f'{metric}_comparison.png')
    plt.savefig(plot_path)
    plt.close()
    print(f"Plot for {metric} saved as {plot_path}")
