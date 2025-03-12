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

# Standardize column names
df_irm.rename(columns={'site_id': 'site'}, inplace=True)
df_lr.rename(columns={'site_left_out': 'site'}, inplace=True)
df_lr.rename(columns={'r2_score': 'r2'}, inplace=True)
df_xgb.rename(columns={'r2_score': 'r2'}, inplace=True)

# Select relevant columns
df_lr = df_lr[['site', 'rmse']].copy()
df_xgb = df_xgb[['site', 'rmse']].copy()
df_lstm = df_lstm[['site', 'rmse']].copy()
df_irm = df_irm[['site', 'rmse']].copy()

# Add model identifiers
df_lr['Model'] = 'LR'
df_xgb['Model'] = 'XGBoost'
df_lstm['Model'] = 'LSTM'
df_irm['Model'] = 'IRM'

# Merge all datasets into a single DataFrame
df_combined = pd.concat([df_lr, df_xgb, df_lstm, df_irm], ignore_index=True)

# Select 20 random unique sites
unique_sites = df_combined['site'].dropna().unique()
selected_sites = np.random.choice(unique_sites, size=10, replace=False)

# Filter data for the selected sites
df_selected = df_combined[df_combined['site'].isin(selected_sites)]

# ===============================
# 2. Plot RMSE trends across models for selected sites
# ===============================
plt.figure(figsize=(12, 6))

# Define x-axis positions for the models, starting with LR
models_order = ["LR", "XGBoost", "LSTM", "IRM"]
x_positions = [1, 2, 3, 4]

for site in selected_sites:
    site_data = df_selected[df_selected["site"] == site]

    # Extract RMSE values for each model in the specified order
    values = [
        site_data.loc[site_data["Model"] == "LR", "rmse"].values,
        site_data.loc[site_data["Model"] == "XGBoost", "rmse"].values,
        site_data.loc[site_data["Model"] == "LSTM", "rmse"].values,
        site_data.loc[site_data["Model"] == "IRM", "rmse"].values
    ]

    # Convert values to a list of floats (skip if any are missing)
    values = [v[0] if len(v) > 0 else np.nan for v in values]
    if any(np.isnan(values)):
        continue  # Skip if there are missing values

    # Plot points and connect them with a line
    plt.scatter(x_positions, values, s=60, label=site if site == selected_sites[0] else "")
    plt.plot(x_positions, values, linestyle='-', marker='o', alpha=0.7)

# Set x-axis labels
plt.xticks(x_positions, models_order)
plt.xlabel("Model")
plt.ylabel("RMSE")
plt.grid(True, linestyle="--", alpha=0.5)

# Save the plot
plt.tight_layout()
plt.savefig("directcomparison_sites_RMSE.png", dpi=300)