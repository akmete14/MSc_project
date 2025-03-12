import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data (Ensure these paths are correct)
df_lr_nostab = pd.read_csv('/cluster/project/math/akmete/MSc/Stabilized_Regression/LOSO/NoStability/LR/results_screened.csv')
df_lr_withstab = pd.read_csv('/cluster/project/math/akmete/MSc/Stabilized_Regression/LOSO/WithStability/LR/results_screened.csv')

# Add labels
df_lr_nostab['Stability'] = 'Screened & Pred-Filtered'
df_lr_withstab['Stability'] = 'Screened & Pred+Stab-Filtered'

# Select relevant columns
df_lr_nostab = df_lr_nostab[['test_site', 'ensemble_rmse_scaled', 'full_rmse_scaled', 'Stability']]
df_lr_withstab = df_lr_withstab[['test_site', 'ensemble_rmse_scaled', 'full_rmse_scaled', 'Stability']]

# Extract the full model RMSE values (same for both)
df_full_rmse = df_lr_nostab[['test_site', 'full_rmse_scaled']].copy()
df_full_rmse.rename(columns={'full_rmse_scaled': 'ensemble_rmse_scaled'}, inplace=True)
df_full_rmse['Stability'] = 'Screened'

# Merge datasets on test_site for comparison
df_merged = df_lr_nostab.merge(df_lr_withstab, on='test_site', suffixes=('_pred', '_predstab'))

# Extract RMSE values
x_vals = df_merged['ensemble_rmse_scaled_pred']  # X-axis: Pred-Filtered
y_vals = df_merged['ensemble_rmse_scaled_predstab']  # Y-axis: Pred+Stab-Filtered

# Get diagonal reference values (Full Model RMSE)
min_val = min(x_vals.min(), y_vals.min())
max_val = max(x_vals.max(), y_vals.max())

# Plot scatter plot
plt.figure(figsize=(12, 7))
sns.scatterplot(x=x_vals, y=y_vals, alpha=0.6)

# Plot the diagonal reference line (Full Model)
plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', label="Full Model (Screened)")

# Labels and title
plt.xlabel("RMSE (Screened & Pred-Filtered)")
plt.ylabel("RMSE (Screened & Pred+Stab-Filtered)")
#plt.title("Comparison of RMSE Values Across Stability Settings")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.5)

# Show the plot
plt.savefig('scatter_LR.png')
