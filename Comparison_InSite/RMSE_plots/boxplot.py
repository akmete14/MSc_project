import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
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
# 2. Combine all models into a single DataFrame
# ===============================
df_combined = pd.concat([df_lr, df_xgb, df_lstm, df_irm], ignore_index=True)

# ===============================
# 3. Filter out values above the 99th quantile for each model
# ===============================
df_filtered = df_combined[
    df_combined['rmse'] <= df_combined.groupby('Model')['rmse'].transform(lambda x: x.quantile(0.98))
]

# ===============================
# 4. Create the boxplot
# ===============================
plt.figure(figsize=(8, 6))
sns.boxplot(x='Model', y='rmse', data=df_filtered)
plt.xlabel("Model")
plt.ylabel("RMSE")
plt.grid(axis="y", linestyle="--", alpha=0.7)

# Save the plot
plt.savefig('boxplot_RMSE.png', dpi=300, bbox_inches='tight')


