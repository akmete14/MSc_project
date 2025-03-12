import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# Read in the data
df_xgb = pd.read_csv('/cluster/project/math/akmete/MSc/XGBoost/LOGO/balanced_grouping/results_balanced_grouping.csv')
df_xgb_weighted = pd.read_csv('/cluster/project/math/akmete/MSc/XGBoost/LOGO/balanced_grouping_weighted_training/results.csv')
#df_lstm = pd.read_csv('/cluster/project/math/akmete/MSc/LSTM/LOGO/results_LOGO.csv')
#df_irm = pd.read_csv('/cluster/project/math/akmete/MSc/IRM/LOGO/LOGO_balanced_grouping/results.csv')
#df_lr = pd.read_csv('/cluster/project/math/akmete/MSc/LR/LOGO/results_2.csv')

# Column names
# XGB: cluster,mse,rmse,r2_score,relative_error,mae
# LR: cluster_left_out,mse,rmse,r2,relative_error,mae
# IRM: cluster,mse,rmse,r2,relative_error,mae
# LSTM: cluster,test_loss,mse,r2,relative_error,mae,rmse
# XGB_weighted: cluster,mse,rmse,r2_score,relative_error,mae

# Standardize column names for the common metrics
df_xgb.rename(columns={'r2_score': 'r2'}, inplace=True)
#df_lr.rename(columns={'cluster_left_out': 'cluster'}, inplace=True)
df_xgb_weighted.rename(columns={'r2_score': 'r2'}, inplace=True)

# Add model identifiers
df_xgb['model'] = 'XGBoost'
df_xgb_weighted['model'] = 'XGBoost Weighted'
#df_lstm['model'] = 'LSTM'
#df_irm['model'] = 'IRM'
#df_lr['model'] = 'LR'

# Select relevant columns
df_xgb = df_xgb[['cluster', 'rmse', 'model']]
#df_lstm = df_lstm[['cluster', 'rmse', 'model']]
#df_irm = df_irm[['cluster', 'rmse', 'model']]
#df_lr = df_lr[['cluster', 'rmse', 'model']]
df_xgb_weighted = df_xgb_weighted[['cluster', 'rmse', 'model']]

# Combine only the columns we need from each dataframe
df_combined = pd.concat([
    df_xgb[['cluster', 'rmse', 'model']],
    df_xgb_weighted[['cluster', 'rmse', 'model']]
    #df_lstm[['cluster', 'rmse', 'model']],
    #df_irm[['cluster', 'rmse', 'model']],
    #df_lr[['cluster', 'rmse', 'model']]
], ignore_index=True)

# Create the barplot
plt.figure(figsize=(12, 6))
sns.barplot(data=df_combined, x='cluster', y='rmse', hue='model')
#plt.title("Comparison of RMSE Across Clusters and Models")
plt.xlabel("Cluster")
plt.ylabel("RMSE")
plt.legend(title="Model")
plt.tight_layout()

# Save the plot (since we're using a non-interactive backend)
plt.savefig("rmse_xgb_weighted.png", dpi=300)
plt.close()
