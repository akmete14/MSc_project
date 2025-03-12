import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

df_lr_nostab = pd.read_csv('/cluster/project/math/akmete/MSc/Stabilized_Regression/In_Site/NoStability/LR/results_lasso_modified.csv')
df_lr_withstab = pd.read_csv('/cluster/project/math/akmete/MSc/Stabilized_Regression/In_Site/WithStability/LR/results_lasso_modified.csv')

# Column names of both dataframes
# no stability: site,ensemble_mse_scaled,full_mse_scaled,ensemble_rmse_scaled,full_rmse_scaled,ensemble_r2,full_r2,ensemble_relative_error,full_relative_error,ensemble_mae,full_mae,O_hat_count
# with stability: site,ensemble_mse_scaled,full_mse_scaled,ensemble_rmse_scaled,full_rmse_scaled,ensemble_r2,full_r2,ensemble_relative_error,full_relative_error,ensemble_mae,full_mae,G_hat_count,O_hat_count,O_hat


# Add labels to identify the datasets
df_lr_nostab['Stability'] = 'Screened & Pred-Filtered'
df_lr_withstab['Stability'] = 'Screened & Pred+Stab-Filtered'

# Select relevant columns
df_lr_nostab = df_lr_nostab[['site', 'ensemble_rmse_scaled', 'full_rmse_scaled', 'Stability']]
df_lr_withstab = df_lr_withstab[['site', 'ensemble_rmse_scaled', 'full_rmse_scaled', 'Stability']]

# Create a third dataset for the full model RMSE (values are the same in both datasets)
df_full_rmse = df_lr_nostab[['site', 'full_rmse_scaled']].copy()
df_full_rmse.rename(columns={'full_rmse_scaled': 'ensemble_rmse_scaled'}, inplace=True)
df_full_rmse['Stability'] = 'Screened'


# Combine all datasets
df_combined = pd.concat([df_lr_nostab, df_lr_withstab, df_full_rmse], ignore_index=True)

# Function to remove upper outliers above the 75th percentile (Q3)
'''def remove_above_q3(df, column='ensemble_rmse_scaled'):
    Q3 = df[column].quantile(0.75)  # 75th percentile
    return df[df[column] <= Q3]  # Keep only values <= Q3

# Apply filtering to remove upper 25% RMSE values
df_combined_filtered = remove_above_q3(df_combined)
'''
# ===============================
# Plot Histogram Comparing Stability Conditions (Without Upper Outliers)
# ===============================
plt.figure(figsize=(12, 6))

sns.histplot(
    data=df_combined,
    x="ensemble_rmse_scaled",  # RMSE values
    hue="Stability",  # Different colors for each stability condition
    kde=False,  # Add kernel density estimate
    alpha=0.5,  # Transparency for better visibility
    bins=20  # Adjust number of bins for better visualization
)

# Formatting axes
plt.xticks(np.arange(df_combined["ensemble_rmse_scaled"].min(), 
                     df_combined["ensemble_rmse_scaled"].max(), 0.01))
plt.ticklabel_format(style='plain', axis='x')

# Labels
plt.xlabel("RMSE (Ensemble Scaled)")
plt.ylabel("Count of Sites")
plt.title("Histogram of RMSE Across Different Feature Selection Strategies")

# Layout adjustments
plt.tight_layout()

# Show the plot
plt.savefig('histo_LR.png')
