import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats

df_xgb_nostab = pd.read_csv('/cluster/project/math/akmete/MSc/Stabilized_Regression/LOGO/NoStability/XGBoost/results_screened.csv')
df_xgb_withstab = pd.read_csv('/cluster/project/math/akmete/MSc/Stabilized_Regression/LOGO/WithStability/XGBoost/results_screened.csv')

# Column names of both dataframes
# no stability: test_cluster,ensemble_mse_scaled,full_mse_scaled,ensemble_rmse_scaled,full_rmse_scaled,ensemble_r2,full_r2,ensemble_relative_error,full_relative_error,ensemble_mae,full_mae,O_hat_count
# with stability: test_cluster,ensemble_mse_scaled,full_mse_scaled,ensemble_rmse_scaled,full_rmse_scaled,ensemble_r2,full_r2,ensemble_relative_error,full_relative_error,ensemble_mae,full_mae,G_hat_count,O_hat_count,O_hat


# Add labels to identify the datasets
df_xgb_nostab['Stability'] = 'Pred'
df_xgb_withstab['Stability'] = 'Pred+Stab'

# Select relevant columns
df_xgb_nostab = df_xgb_nostab[['test_cluster', 'ensemble_rmse_scaled', 'full_rmse_scaled', 'Stability']]
df_xgb_withstab = df_xgb_withstab[['test_cluster', 'ensemble_rmse_scaled', 'full_rmse_scaled', 'Stability']]

# Create a third dataset for the full model RMSE (values are the same in both datasets)
df_full_rmse = df_xgb_nostab[['test_cluster', 'full_rmse_scaled']].copy()
df_full_rmse.rename(columns={'full_rmse_scaled': 'ensemble_rmse_scaled'}, inplace=True)
df_full_rmse['Stability'] = 'Full'
'''
# Function to remove the top 1% outliers
def filter_rmse_99(df, column='ensemble_rmse_scaled'):
    q_high = df[column].quantile(0.99)
    return df[df[column] <= q_high]

# Apply filtering to remove upper 1% RMSE values
df_xgb_nostab = filter_rmse_99(df_xgb_nostab)
df_xgb_withstab = filter_rmse_99(df_xgb_withstab)
df_full_rmse = filter_rmse_99(df_full_rmse, column='ensemble_rmse_scaled')
'''
# Combine all datasets after filtering
df_combined_filtered = pd.concat([df_xgb_nostab, df_xgb_withstab, df_full_rmse], ignore_index=True)

# Create a boxplot comparing RMSE scaled values across conditions after filtering
plt.figure(figsize=(8, 6))
sns.boxplot(x='Stability', y='ensemble_rmse_scaled', data=df_combined_filtered)
plt.xlabel("Model Type")
plt.ylabel("RMSE")
#plt.title("Comparison of RMSE Scaled Across Different Feature Selection Strategies")
plt.xticks(rotation=15)  # Rotate labels for better readability
plt.grid(axis='y', linestyle='--', alpha=0.7)
# Show the plot
plt.savefig('boxplot_logo_XGB.png')

# Perform statistical t-tests between the filtered groups
t_stat_nostab_full, p_value_nostab_full = stats.ttest_ind(df_xgb_nostab['ensemble_rmse_scaled'], df_full_rmse['ensemble_rmse_scaled'], equal_var=False)
t_stat_withstab_full, p_value_withstab_full = stats.ttest_ind(df_xgb_withstab['ensemble_rmse_scaled'], df_full_rmse['ensemble_rmse_scaled'], equal_var=False)

# Print the t-test results
print(f"T-test (No Stability vs. Full Model, filtered): T-statistic = {t_stat_nostab_full:.4f}, P-value = {p_value_nostab_full:.4f}")
print(f"T-test (With Stability vs. Full Model, filtered): T-statistic = {t_stat_withstab_full:.4f}, P-value = {p_value_withstab_full:.4f}")