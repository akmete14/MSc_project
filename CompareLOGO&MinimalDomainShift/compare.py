import pandas as pd
import numpy as np

df_logo = pd.read_csv('/cluster/project/math/akmete/MSc/LOGO/LOGOCVXGB_100hrs.csv')
df_mds = pd.read_csv('/cluster/project/math/akmete/MSc/Minimal_DomainShift_Extrapolation/Metrics_MinimalDomainShift_XGB.csv')
df_mds = df_mds.drop(columns=['Unnamed: 0', 'num_samples'])

df_logo['rmse'] = np.sqrt(df_logo['mse'])
df_mds['rmse'] = np.sqrt(df_mds['mse'])

print(df_logo)
print(df_mds)

import matplotlib.pyplot as plt
import os

# Merge the DataFrames with different column names
comparison_df = pd.merge(
    df_logo.rename(columns={'group_left_out': 'group'}),  # Rename for consistency
    df_mds.rename(columns={'cluster': 'group'}),    # Rename for consistency
    on='group',
    suffixes=('_left_out', '_minimal_shift')
)

# Plot the comparison as a bar chart
comparison_df.plot(
    x='group',
    y=['rmse_left_out', 'rmse_minimal_shift'],
    kind='bar',
    figsize=(12, 6),
    title='RMSE Comparison for Each Group Across Settings'
)

# Customize the plot
plt.ylabel('RMSE')
plt.xlabel('Cluster')
plt.xticks(rotation=45, ha='right')  # Rotate group labels for better readability
plt.legend(['LOGO Setting', 'Minimal Domain Shift Setting'])
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()  # Adjust layout to prevent label overlap

# Save the figure in the current directory
output_path = os.path.join(os.getcwd(), "rmse_comparison_plot.png")
plt.savefig(output_path, dpi=300)  # Save with high resolution
plt.close()  # Close the plot to free memory

print(f"Plot saved at: {output_path}")

import matplotlib.pyplot as plt
import os

# Prepare the data for the boxplot
data = [
    comparison_df['rmse_left_out'],
    comparison_df['rmse_minimal_shift']
]

# Create the boxplot
plt.figure(figsize=(8, 6))
plt.boxplot(data, labels=['LOGO Setting', 'Minimal Domain Shift Setting'])

# Customize the plot
plt.title('RMSE Distribution LOGO vs Minimal Domain Shift Setting')
plt.ylabel('RMSE')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()

# Save the boxplot in the current directory
output_path = os.path.join(os.getcwd(), "rmse_boxplot.png")
plt.savefig(output_path, dpi=300)
plt.close()  # Close the plot to free memory

print(f"Boxplot saved at: {output_path}")


