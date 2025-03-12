import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# Read in the data
df_xgb = pd.read_csv('/cluster/project/math/akmete/MSc/XGBoost/LOSO/results.csv')
df_lstm = pd.read_csv('/cluster/project/math/akmete/MSc/LSTM/LOSO/results.csv')
df_irm = pd.read_csv('/cluster/project/math/akmete/MSc/IRM/LOSO/results_global_scaling/results.csv')
df_lr = pd.read_csv('/cluster/project/math/akmete/MSc/LR/LOSO/results_2.csv')

# Column names
# XGB: site,mse,rmse,r2_score,relative_error,mae
# LR: site_left_out,mse,rmse,r2,relative_error,mae
# IRM: site_id,mse,rmse,r2,relative_error,mae
# LSTM: site,test_loss,mse,r2,relative_error,mae,rmse

# Standardize column names for the common metrics
df_xgb.rename(columns={'r2_score': 'r2'}, inplace=True)
df_irm.rename(columns={'site_id': 'site'}, inplace=True)
df_lr.rename(columns={'site_left_out': 'site'}, inplace=True)

# Add model identifiers
df_xgb['model'] = 'XGBoost'
df_lstm['model'] = 'LSTM'
df_irm['model'] = 'IRM'
df_lr['model'] = 'LR'

# Select relevant columns
df_xgb = df_xgb[['site', 'rmse', 'model']]
df_lstm = df_lstm[['site', 'rmse', 'model']]
df_irm = df_irm[['site', 'rmse', 'model']]
df_lr = df_lr[['site', 'rmse', 'model']]

# Function to remove upper outliers above the 99th percentile
def remove_upper_outliers(df, column='rmse', threshold=0.99):
    q_high = df[column].quantile(threshold)
    return df[df[column] <= q_high]

# Apply outlier removal
#df_xgb = remove_upper_outliers(df_xgb)
df_lstm = remove_upper_outliers(df_lstm)
#df_irm = remove_upper_outliers(df_irm)
#df_lr = remove_upper_outliers(df_lr)

# Combine all datasets after removing outliers
df_combined_filtered = pd.concat([df_xgb, df_lstm, df_irm, df_lr], ignore_index=True)

# Create a boxplot comparing RMSE for different models (after removing upper 5% outliers)
plt.figure(figsize=(8, 6))
sns.boxplot(x='model', y='rmse', data=df_combined_filtered)
plt.xlabel("Model")
plt.ylabel("RMSE")
#plt.title("Comparison of RMSE Across Models (Without Upper 5% Outliers)")
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Layout adjustments
plt.tight_layout()

plt.savefig('boxplot_rmse_loso.png', dpi=300)