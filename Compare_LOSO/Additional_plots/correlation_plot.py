# Import libraries
import pandas as pd
import matplotlib.pyplot as plt

# Read data
df_metrics = pd.read_csv('/cluster/project/math/akmete/MSc/XGBoost/LOSO/results.csv')
df_targets = pd.read_csv('/cluster/project/math/akmete/MSc/preprocessing/df_balanced_groups_onevegindex.csv')


# Group target data by site_id to compute mean & std
df_site_stats = (df_targets
                 .groupby('site_id')['GPP']
                 .agg(['mean', 'std'])
                 .reset_index()
                 .rename(columns={'mean': 'target_mean', 'std': 'target_std'}))

# Merge df_metrics with df_site_stats
df_merged = pd.merge(df_metrics, df_site_stats,
                     left_on='site', right_on='site_id',
                     how='inner')

# Drop site_id column
df_merged.drop(columns=['site_id'], inplace=True)

# Visualize RMSE vs. target_mean and RMSE vs target_std
plt.figure(figsize=(8,6))
plt.scatter(df_merged['target_mean'], df_merged['rmse'], color='blue', label='Mean vs RMSE')
plt.scatter(df_merged['target_std'],  df_merged['rmse'], color='orange', label='Std vs RMSE')
plt.legend()

# Rename x-axis to clarify it’s showing two different stats
plt.xlabel('Target Mean (Blue) / Std (Orange)')
plt.ylabel('RMSE')
plt.tight_layout()
#plt.title('RMSE vs. Target Mean and Std')
plt.savefig('RMSEvsTargetStd.png', dpi=300)