import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# --- Read in the data ---
# In-Site files
df_xgb_insite_nostab = pd.read_csv('/cluster/project/math/akmete/MSc/Stabilized_Regression/In_Site/NoStability/XGBoost/results_screened_O_hat_modified_top7.csv')
df_xgb_insite_withstab = pd.read_csv('/cluster/project/math/akmete/MSc/Stabilized_Regression/In_Site/WithStability/XGBoost/results_screened_all_metrics_modified.csv')

# LOSO files
df_xgb_loso_nostab = pd.read_csv('/cluster/project/math/akmete/MSc/Stabilized_Regression/LOSO/NoStability/XGBoost/plots_and_results_50balanced/results_50sites.csv')
df_xgb_loso_withstab = pd.read_csv('/cluster/project/math/akmete/MSc/Stabilized_Regression/LOSO/WithStability/XGBoost/plots_and_results_50balanced/results_50sites.csv')

# LOGO files
df_xgb_logo_nostab = pd.read_csv('/cluster/project/math/akmete/MSc/Stabilized_Regression/LOGO/NoStability/XGBoost/results_screened.csv')
df_xgb_logo_withstab = pd.read_csv('/cluster/project/math/akmete/MSc/Stabilized_Regression/LOGO/WithStability/XGBoost/results_screened.csv')

# --- Rename identifier columns if needed ---
# For LOGO files, rename identifier column to 'site'
df_xgb_logo_nostab.rename(columns={'test_cluster': 'site'}, inplace=True)
df_xgb_logo_withstab.rename(columns={'test_cluster': 'site'}, inplace=True)
# For LOSO files, rename identifier column to 'site'
df_xgb_loso_nostab.rename(columns={'test_site': 'site'}, inplace=True)
df_xgb_loso_withstab.rename(columns={'test_site': 'site'}, inplace=True)

# --- Create a function to extract and label data for each setting ---
def create_setting_df(df_nostab, df_withstab, setting_name):
    # Full Model: use the full_rmse_scaled column (identical in both files)
    df_full = df_nostab[['site', 'full_rmse_scaled']].copy()
    df_full.rename(columns={'full_rmse_scaled': 'rmse'}, inplace=True)
    df_full['model'] = 'Full Model'
    df_full['setting'] = setting_name
    
    # Pred Model: from NoStability file's rmse
    df_pred = df_nostab[['site', 'ensemble_rmse_scaled']].copy()
    df_pred.rename(columns={'ensemble_rmse_scaled':'rmse'}, inplace=True)
    df_pred['model'] = 'Pred Model'
    df_pred['setting'] = setting_name
    
    # Pred+Stab Model: from WithStability file's rmse
    df_predstab = df_withstab[['site', 'ensemble_rmse_scaled']].copy()
    df_predstab.rename(columns={'ensemble_rmse_scaled':'rmse'}, inplace=True)
    df_predstab['model'] = 'Pred+Stab Model'
    df_predstab['setting'] = setting_name
    
    return pd.concat([df_full, df_pred, df_predstab], axis=0)

# --- Build dataframes for each setting ---
df_insite = create_setting_df(df_xgb_insite_nostab, df_xgb_insite_withstab, 'In-Site')
df_loso   = create_setting_df(df_xgb_loso_nostab, df_xgb_loso_withstab, 'LOSO')
df_logo   = create_setting_df(df_xgb_logo_nostab, df_xgb_logo_withstab, 'LOGO')

# Define a threshold for the full model in the In-Site setting
#threshold_full_insite = 1  # adjust this value as needed

# Filter the In-Site DataFrame: keep only rows for the Full Model with RMSE below the threshold
#df_insite = df_insite[~((df_insite['model'] == 'Full Model') & (df_insite['rmse'] > threshold_full_insite))]

# --- Combine all settings ---
df_combined = pd.concat([df_insite, df_loso, df_logo], ignore_index=True)

# --- (Optional) Remove outliers if needed ---
# threshold = some_value
# df_combined = df_combined[df_combined['rmse'] < threshold]

# --- Compute summary statistics for each setting and model ---
stats = df_combined.groupby(['setting', 'model'])['rmse'].agg(['mean', 'std']).reset_index()

# --- Plotting ---
settings_order = ['In-Site', 'LOSO', 'LOGO']
models_order = ['Full Model', 'Pred Model', 'Pred+Stab Model']

x_positions = np.arange(len(settings_order))
n_models = len(models_order)
width = 0.1

fig, ax = plt.subplots(figsize=(6, 6))
for i, model in enumerate(models_order):
    data_model = stats[stats['model'] == model]
    means, stds, xs = [], [], []
    for j, setting in enumerate(settings_order):
        row = data_model[data_model['setting'] == setting]
        if not row.empty:
            means.append(row['mean'].values[0])
            stds.append(row['std'].values[0])
            xs.append(x_positions[j] + (i - n_models/2 + 0.5)*width)
    ax.errorbar(xs, means, yerr=stds, fmt='o', capsize=5, markersize=8, label=model)

ax.set_xticks(x_positions)
ax.set_xticklabels(settings_order)
ax.set_xlabel("Setting")
ax.set_ylabel("RMSE")
ax.legend(title="Model")
plt.tight_layout()
plt.savefig("averagermse_SR_XGB.png", dpi=300)
plt.close()
