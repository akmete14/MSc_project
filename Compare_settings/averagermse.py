import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# Read in the data
df_xgb_insite = pd.read_csv('/cluster/project/math/akmete/MSc/XGBoost/InSite/results_2.csv')
df_xgb_loso = pd.read_csv('/cluster/project/math/akmete/MSc/XGBoost/LOSO/results.csv')
df_xgb_logo = pd.read_csv('/cluster/project/math/akmete/MSc/XGBoost/LOGO/balanced_grouping/results_balanced_grouping.csv')


# Label each dataset with its setting and model
df_xgb_insite['setting'] = 'In-Site'
df_xgb_loso['setting']   = 'LOSO'
df_xgb_logo['setting']   = 'LOGO'
df_xgb_insite['model']   = 'XGBoost'
df_xgb_loso['model']     = 'XGBoost'
df_xgb_logo['model']     = 'XGBoost'
# For consistency, rename LOGO's identifier column if needed:
df_xgb_logo.rename(columns={'cluster': 'site'}, inplace=True)

# Column names:
# insite: site,mse,rmse,r2_score,relative_error,mae
# loso: site,mse,rmse,r2_score,relative_error,mae
# logo: cluster,mse,rmse,r2_score,relative_error,mae

df_lstm_insite = pd.read_csv('/cluster/project/math/akmete/MSc/LSTM/InSite/results_modified.csv')
df_lstm_loso = pd.read_csv('/cluster/project/math/akmete/MSc/LSTM/LOSO/results.csv')
df_lstm_logo = pd.read_csv('/cluster/project/math/akmete/MSc/LSTM/LOGO/results_LOGO_modified.csv')

# Column names:
# insite: site,test_loss,mse,r2,relative_error,mae,rmse
# loso: site,test_loss,mse,r2,relative_error,mae,rmse
# logo: cluster,test_loss,mse,r2,relative_error,mae,rmse

# Label with setting and model name
df_lstm_insite['setting'] = 'In-Site'
df_lstm_loso['setting']   = 'LOSO'
df_lstm_logo['setting']   = 'LOGO'
df_lstm_insite['model']   = 'LSTM'
df_lstm_loso['model']     = 'LSTM'
df_lstm_logo['model']     = 'LSTM'
# Rename LOGO identifier if needed
df_lstm_logo.rename(columns={'cluster': 'site'}, inplace=True)

# Add linear regression
df_lr_insite = pd.read_csv('/cluster/project/math/akmete/MSc/LR/In_Site/results_2.csv')
df_lr_loso = pd.read_csv('/cluster/project/math/akmete/MSc/LR/LOSO/results_2.csv')
df_lr_logo = pd.read_csv('/cluster/project/math/akmete/MSc/LR/LOGO/results_2.csv')

# Column names:
# insite: site,mse,rmse,r2_score,relative_error,mae
# loso: site_left_out,mse,rmse,r2,relative_error,mae
# logo: cluster_left_out,mse,rmse,r2,relative_error,mae

# Label with setting and model name
df_lr_insite['setting'] = 'In-Site'
df_lr_loso['setting']   = 'LOSO'
df_lr_logo['setting']   = 'LOGO'
df_lr_insite['model']   = 'LR'
df_lr_loso['model']     = 'LR'
df_lr_logo['model']     = 'LR'
# Rename LOGO identifier if needed
df_lr_logo.rename(columns={'cluster_left_out': 'site'}, inplace=True)
df_lr_loso.rename(columns={'site_left_out': 'site'}, inplace=True)

# Remone one large outlier of LR insite
threshold = 100  # Adjust this value as needed
df_lr_insite = df_lr_insite[df_lr_insite['rmse'] < threshold]

# Add IRM:
df_irm_insite = pd.read_csv('/cluster/project/math/akmete/MSc/IRM/OnSite/results_corrected.csv')
df_irm_loso = pd.read_csv('/cluster/project/math/akmete/MSc/IRM/LOSO/results_global_scaling/results.csv')
df_irm_logo = pd.read_csv('/cluster/project/math/akmete/MSc/IRM/LOGO/LOGO_balanced_grouping/results.csv')

# Column names:
# insite: site_id,mse,rmse,r2,relative_error,mae
# loso: site_id,mse,rmse,r2,relative_error,mae
# logo: cluster,mse,rmse,r2,relative_error,mae

# Label with setting and model name
df_irm_insite['setting'] = 'In-Site'
df_irm_loso['setting']   = 'LOSO'
df_irm_logo['setting']   = 'LOGO'
df_irm_insite['model']   = 'IRM'
df_irm_loso['model']     = 'IRM'
df_irm_logo['model']     = 'IRM'
# Rename LOGO identifier if needed
df_irm_logo.rename(columns={'cluster': 'site'}, inplace=True)
df_irm_loso.rename(columns={'site_id': 'site'}, inplace=True)
df_irm_insite.rename(columns={'site_id': 'site'}, inplace=True)

#threshold
threshold_irm = 1
df_irm_insite = df_irm_insite[df_irm_insite['rmse'] < threshold_irm]



df_combined = pd.concat([
    df_xgb_insite[['site', 'rmse', 'setting', 'model']],
    df_xgb_loso[['site', 'rmse', 'setting', 'model']],
    df_xgb_logo[['site', 'rmse', 'setting', 'model']],
    df_lstm_insite[['site', 'rmse', 'setting', 'model']],
    df_lstm_loso[['site', 'rmse', 'setting', 'model']],
    df_lstm_logo[['site', 'rmse', 'setting', 'model']],
    df_lr_insite[['site', 'rmse', 'setting', 'model']],
    df_lr_loso[['site', 'rmse', 'setting', 'model']],
    df_lr_logo[['site', 'rmse', 'setting', 'model']],
    df_irm_insite[['site', 'rmse', 'setting', 'model']],
    df_irm_loso[['site', 'rmse', 'setting', 'model']],
    df_irm_logo[['site', 'rmse', 'setting', 'model']]
], ignore_index=True)

# Compute summary statistics for each setting and model
stats = df_combined.groupby(['setting', 'model'])['rmse'].agg(['mean', 'std']).reset_index()

# Define the order of settings for the x-axis
settings_order = ['In-Site', 'LOSO', 'LOGO']
# Define the models (e.g., XGBoost and LSTM) in a fixed order:
models_order = stats['model'].unique()

# Create x positions for each setting
x_positions = np.arange(len(settings_order))
n_models = len(models_order)
# Set the horizontal offset for each model (adjust width as needed)
width = 0.1

fig, ax = plt.subplots(figsize=(6, 6))

# Loop over each model and plot its point with error bars
for i, model in enumerate(models_order):
    # Filter the stats for this model
    data_model = stats[stats['model'] == model]
    means, stds, xs = [], [], []
    for j, setting in enumerate(settings_order):
        row = data_model[data_model['setting'] == setting]
        if not row.empty:
            means.append(row['mean'].values[0])
            stds.append(row['std'].values[0])
            # Calculate the x position with an offset based on the model index
            xs.append(x_positions[j] + (i - n_models/2 + 0.5) * width)
    ax.errorbar(xs, means, yerr=stds, fmt='o', capsize=5, markersize=8, label=model)

ax.set_xticks(x_positions)
ax.set_xticklabels(settings_order)
ax.set_xlabel("Setting")
ax.set_ylabel("RMSE")
#ax.set_title("Average RMSE with Error Bars Across Settings for Two Models")
ax.legend(title="Model")
plt.tight_layout()
plt.savefig("averagermse.png", dpi=300)
plt.close()