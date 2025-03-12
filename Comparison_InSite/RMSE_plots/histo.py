import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# ===============================
# 1. Read CSV files and standardize columns
# ===============================
df_lr = pd.read_csv('/cluster/project/math/akmete/MSc/LR/In_Site/results_2.csv')
df_xgb = pd.read_csv('/cluster/project/math/akmete/MSc/XGBoost/InSite/results_2.csv')
df_lstm = pd.read_csv('/cluster/project/math/akmete/MSc/LSTM/InSite/results_modified.csv')
df_irm = pd.read_csv('/cluster/project/math/akmete/MSc/IRM/OnSite/results_corrected.csv')

# Standardize column names
df_irm.rename(columns={'site_id': 'site'}, inplace=True)
df_lr.rename(columns={'r2_score': 'r2'}, inplace=True)
df_xgb.rename(columns={'r2_score': 'r2'}, inplace=True)

# Add model identifiers
df_lr['Model'] = 'LR'
df_xgb['Model'] = 'XGBoost'
df_lstm['Model'] = 'LSTM'
df_irm['Model'] = 'IRM'

# Select relevant columns
df_lr = df_lr[['Model', 'rmse']]
df_xgb = df_xgb[['Model', 'rmse']]
df_lstm = df_lstm[['Model', 'rmse']]
df_irm = df_irm[['Model', 'rmse']]

# ===============================
# 2. Remove upper outliers using the 98th percentile
# ===============================
def remove_above_q98(df, column='rmse'):
    q98 = df[column].quantile(0.98)  # 98th percentile
    return df[df[column] <= q98]     # Keep only values <= q98

# Apply filtering to all models
df_lr_filtered = remove_above_q98(df_lr)
df_irm_filtered = remove_above_q98(df_irm)
df_xgb_filtered = remove_above_q98(df_xgb)
df_lstm_filtered = remove_above_q98(df_lstm)

# Combine all models into a single DataFrame
df_combined = pd.concat([df_lr_filtered, df_xgb_filtered, df_lstm_filtered, df_irm_filtered], ignore_index=True)

# ===============================
# 3. Plot Histogram Comparing Models (Without Upper Outliers)
# ===============================
plt.figure(figsize=(8, 6))

sns.histplot(
    data=df_combined,
    x="rmse",      # RMSE values
    hue="Model",   # Different colors for each model
    kde=True,      # Add kernel density estimate
    alpha=0.4,     # Transparency for better visibility
    bins=60       # Adjust the number of bins for better visualization
)

# Formatting axes
plt.xlim(0, 0.085)  # Adjust x-axis range
plt.xticks(np.arange(0, 0.1, 0.01))
plt.ticklabel_format(style='plain', axis='x')

# Labels
plt.xlabel("RMSE")
plt.ylabel("Count of Sites")

# Layout adjustments
plt.tight_layout()

# Save the plot
plt.savefig("histo_RMSE_98quantile.png", dpi=300, bbox_inches='tight')

