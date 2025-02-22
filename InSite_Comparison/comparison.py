import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

# Read CSV files
df_lr = pd.read_csv('/cluster/project/math/akmete/MSc/LR/In_Site/results.csv')
df_xgb = pd.read_csv('/cluster/project/math/akmete/MSc/XGBoost/InSite/results.csv') 
df_lstm = pd.read_csv('/cluster/project/math/akmete/MSc/XGBvsLSTM_OnSite/LSTM_extrapolation_in_time_single_site.csv')
df_irm = pd.read_csv('/cluster/project/math/akmete/MSc/IRM/OnSite/results.csv')
df_sr_nostab = pd.read_csv('/cluster/project/math/akmete/MSc/Stabilized_Regression/In_Site/NoStability/results.csv')
df_sr_withstab = pd.read_csv('/cluster/project/math/akmete/MSc/Stabilized_Regression/In_Site/WithStability/results.csv')

# Remove outlier from LR using the IQR method on the 'rmse' column
q1_lr = df_lr['rmse'].quantile(0.25)
q3_lr = df_lr['rmse'].quantile(0.75)
iqr_lr = q3_lr - q1_lr
lower_bound_lr = q1_lr - 1.5 * iqr_lr
upper_bound_lr = q3_lr + 1.5 * iqr_lr
df_lr_filtered = df_lr[(df_lr['rmse'] >= lower_bound_lr) & (df_lr['rmse'] <= upper_bound_lr)]

# Remove outliers in IRM using the IQR method on the 'rmse' column
q1_irm = df_irm['rmse'].quantile(0.25)
q3_irm = df_irm['rmse'].quantile(0.75)
iqr_irm = q3_irm - q1_irm
lower_bound_irm = q1_irm - 1.5 * iqr_irm
upper_bound_irm = q3_irm + 1.5 * iqr_irm
df_irm_filtered = df_irm[(df_irm['rmse'] >= lower_bound_irm) & (df_irm['rmse'] <= upper_bound_irm)]

# Combine RMSE values into one DataFrame with corresponding model names
# For Stabilized Regression, we use the 'full_rmse' column
rmse_data = pd.DataFrame({
    'Model': (['Linear Regression'] * len(df_lr_filtered) +
              ['XGBoost'] * len(df_xgb) +
              ['LSTM'] * len(df_lstm) +
              ['IRM'] * len(df_irm_filtered) +
              ['Stabilized Regression (No Stability)'] * len(df_sr_nostab) +
              ['Stabilized Regression (With Stability)'] * len(df_sr_withstab)),
    'RMSE': pd.concat([
              df_lr_filtered['rmse'],
              df_xgb['rmse'],
              df_lstm['rmse'],
              df_irm_filtered['rmse'],
              df_sr_nostab['full_rmse'],
              df_sr_withstab['full_rmse']
          ], ignore_index=True)
})

# Create a boxplot comparing RMSE across models
plt.figure(figsize=(12, 6))
sns.boxplot(x='Model', y='RMSE', data=rmse_data)
plt.xticks(rotation=45, ha='right')
plt.title('RMSE Comparison Across Different Models')
plt.xlabel('Model')
plt.ylabel('RMSE')
plt.tight_layout()
plt.savefig('boxplot_in_site.png')


# df_lr columns are: site,mse,rmse,r2_score
# df_xgb columns are: site,mse,rmse,r2_score
# df_lstm columns are: site,test_loss,mse,r2,relative_error,mae,rmse
# df_irm columns are: site_id,mse,rmse
# df_sr_nostab columns are: site,ensemble_mse_scaled,full_mse_scaled,O_hat_count,ensemble_rmse,full_rmse
# df_sr_withstab columns are: site,ensemble_mse_scaled,full_mse_scaled,G_hat_count,O_hat_count,O_hat,ensemble_rmse,full_rmse