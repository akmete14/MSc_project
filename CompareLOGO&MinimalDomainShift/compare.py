import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

df_logo = pd.read_csv('/cluster/project/math/akmete/MSc/LOGO/LOGOCVXGB_all_metrics.csv')
df_mds = pd.read_csv('/cluster/project/math/akmete/MSc/Minimal_DomainShift_Extrapolation/AllMetrics_MinimalDomainShift_XGB.csv')
df_mds = df_mds.drop(columns=['num_samples'])

# Compute RMSE for consistency
df_logo['RMSE'] = np.sqrt(df_logo['mse'])
df_mds['RMSE'] = np.sqrt(df_mds['mse'])

# Rename columns for consistency
df_logo = df_logo.rename(columns={
    'group_left_out': 'group', 
    'R2': 'r2', 
    'Relative Error': 'RE'
})
df_mds = df_mds.rename(columns={'cluster': 'group'})

# Merge the DataFrames
comparison_df = pd.merge(
    df_logo,
    df_mds,
    on='group',
    suffixes=('_left_out', '_minimal_shift')
)

# Debugging: Print the columns to verify
print("Columns in comparison_df:", comparison_df.columns)

# Create folder for saving plots
output_folder = os.path.join(os.getcwd(), "plots_comparison")
os.makedirs(output_folder, exist_ok=True)

# Define the metrics to compare
metrics = ['mse', 'RMSE', 'r2', 'RE', 'MAE']

# Create bar plots for each metric
for metric in metrics:
    comparison_df.plot(
        x='group',
        y=[f'{metric}_left_out', f'{metric}_minimal_shift'],
        kind='bar',
        figsize=(12, 6),
        title=f'{metric.upper()} Comparison for Each Group Across Settings'
    )
    plt.ylabel(metric.upper())
    plt.xlabel('Cluster')
    plt.xticks(rotation=45, ha='right')  # Rotate group labels for better readability
    plt.legend(['LOGO Setting', 'Minimal Domain Shift Setting'])
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()

    # Save the bar plot
    output_path = os.path.join(output_folder, f"{metric.lower()}_barplot.png")
    plt.savefig(output_path, dpi=300)
    plt.close()  # Close the plot to free memory
    print(f"{metric.upper()} bar plot saved at: {output_path}")

# Combined RMSE Boxplot
plt.figure(figsize=(8, 6))

# Add data for the two settings
data = [
    comparison_df['RMSE_left_out'],
    comparison_df['RMSE_minimal_shift']
]

# Create the combined RMSE boxplot
plt.boxplot(data, labels=['LOGO Setting', 'Minimal Domain Shift Setting'])
plt.title('RMSE Comparison Between LOGO and Minimal Domain Shift Settings')
plt.ylabel('RMSE')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()

# Save the combined RMSE boxplot
output_path = os.path.join(output_folder, "rmse_combined_boxplot.png")
plt.savefig(output_path, dpi=300)
plt.close()  # Close the plot to free memory
print(f"Combined RMSE boxplot saved at: {output_path}")

