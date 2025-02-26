import os
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

# File paths (update these paths as needed)
csv_no_stability = '/cluster/project/math/akmete/MSc/Stabilized_Regression/In_Site/NoStability/LR/results_lasso.csv'
csv_stability = '/cluster/project/math/akmete/MSc/Stabilized_Regression/In_Site/WithStability/LR/results_lasso.csv'

# Load the CSV files into DataFrames
df_no_stability = pd.read_csv(csv_no_stability)
df_stability = pd.read_csv(csv_stability)
print(df_no_stability['site'].head())
print(df_stability['site'].head())

# Drop columns that are available in the stability DataFrame but not in the no_stability DataFrame
df_stability = df_stability.drop(columns=['G_hat_count', 'O_hat'])

# Merge the DataFrames on the 'site' column; add suffixes to distinguish between the two versions
merged_df = df_no_stability.merge(df_stability, on='site', suffixes=('_no_stab', '_stab'))

# Define the ensemble metric columns to compare
ensemble_metrics = [
    'ensemble_mse_scaled',
    'ensemble_rmse_scaled',
    'ensemble_r2',
    'ensemble_relative_error',
    'ensemble_mae'
]

# Create a folder to save the plots if it doesn't exist
plots_folder = "plots_LR"
os.makedirs(plots_folder, exist_ok=True)

# Loop through each metric, remove outliers, and create a scatter plot
for metric in ensemble_metrics:
    x_metric = f"{metric}_no_stab"
    y_metric = f"{metric}_stab"
    
    # Calculate the IQR for the no_stability column
    Q1_x = merged_df[x_metric].quantile(0.25)
    Q3_x = merged_df[x_metric].quantile(0.75)
    IQR_x = Q3_x - Q1_x
    lower_bound_x = Q1_x - 1.5 * IQR_x
    upper_bound_x = Q3_x + 1.5 * IQR_x
    
    # Calculate the IQR for the with_stability column
    Q1_y = merged_df[y_metric].quantile(0.25)
    Q3_y = merged_df[y_metric].quantile(0.75)
    IQR_y = Q3_y - Q1_y
    lower_bound_y = Q1_y - 1.5 * IQR_y
    upper_bound_y = Q3_y + 1.5 * IQR_y
    
    # Filter rows where both columns fall within the computed bounds
    filtered_df = merged_df[(merged_df[x_metric] >= lower_bound_x) & (merged_df[x_metric] <= upper_bound_x) &
                              (merged_df[y_metric] >= lower_bound_y) & (merged_df[y_metric] <= upper_bound_y)]
    
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=filtered_df, x=x_metric, y=y_metric)
    
    # Plot the line of equality (45° line)
    line_min = min(filtered_df[x_metric].min(), filtered_df[y_metric].min())
    line_max = max(filtered_df[x_metric].max(), filtered_df[y_metric].max())
    plt.plot([line_min, line_max], [line_min, line_max], ls="--", color="red")
    
    plt.title(f"Comparison of {metric.replace('_', ' ').title()} (Outliers Removed)")
    plt.xlabel("No Stability")
    plt.ylabel("With Stability")
    plt.grid(True)
    
    # Save the figure to the specified folder
    plot_filename = os.path.join(plots_folder, f"{metric}_comparison_no_outliers.png")
    plt.savefig(plot_filename, bbox_inches='tight')
    plt.close()
