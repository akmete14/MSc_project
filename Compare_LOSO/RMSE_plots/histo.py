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
df_xgb['Model'] = 'XGBoost'
df_lstm['Model'] = 'LSTM'
df_irm['Model'] = 'IRM'
df_lr['Model'] = 'LR'

# Select relevant columns
df_lr = df_lr[['Model', 'rmse']]
df_xgb = df_xgb[['Model', 'rmse']]
df_lstm = df_lstm[['Model', 'rmse']]
df_irm = df_irm[['Model', 'rmse']]

# Function to remove upper outliers above the 75th percentile (Q3)
def remove_above_q3(df, column='rmse'):
    Q3 = df[column].quantile(0.99)  # 99th percentile
    return df[df[column] <= Q3]  # Keep only values <= Q3

# Apply filtering to all models
df_lr_filtered = remove_above_q3(df_lr)
df_irm_filtered = remove_above_q3(df_irm)
df_xgb_filtered = remove_above_q3(df_xgb)
df_lstm_filtered = remove_above_q3(df_lstm)

# Combine all models into a single DataFrame
df_combined = pd.concat([df_lr, df_xgb, df_lstm_filtered, df_irm], ignore_index=True)

# ===============================
# 2. Plot Histogram Comparing Models (Without Upper Outliers)
# ===============================
plt.figure(figsize=(8, 6))

sns.histplot(
    data=df_combined,
    x="rmse",   # RMSE values
    hue="Model",  # Different colors for each model
    kde=True,  # Add kernel density estimate
    alpha=0.4,  # Transparency for better visibility
    bins=40  # Adjust the number of bins for better visualization
)

# Formatting axes
plt.xlim(0, 0.07)  # Adjust x-axis range
plt.xticks(np.arange(0, 0.1, 0.01))
plt.ticklabel_format(style='plain', axis='x')

# Labels
plt.xlabel("RMSE")
plt.ylabel("Count of Sites")

# Layout adjustments
plt.tight_layout()

# Save the plot
plt.savefig("histogram_rmse_filteredlstm_99.png", dpi=300)

