import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Load CSV files
xgboost_df = pd.read_csv('xgb_extrapolation_in_time_single_site.csv')
lstm_df = pd.read_csv('LSTM_extrapolation_in_time_single_site.csv')
irm_df = pd.read_csv('/cluster/project/math/akmete/MSc/IRM/OnSite/results.csv')

# Remove outliers of xgb lstm and irm
#XGB
q1 = xgboost_df['rmse'].quantile(0.25)
q3 = xgboost_df['rmse'].quantile(0.75)
iqr = q3 - q1

lower_bound = q1 - 1.5 * iqr
upper_bound = q3 + 1.5 * iqr

filtered_xgboost_df = xgboost_df[(xgboost_df['rmse'] >= lower_bound) & (xgboost_df['rmse'] <= upper_bound)]

#LSTM
q1 = lstm_df['rmse'].quantile(0.25)
q3 = lstm_df['rmse'].quantile(0.75)
iqr = q3 - q1

lower_bound = q1 - 1.5 * iqr
upper_bound = q3 + 1.5 * iqr

filtered_lstm_df = lstm_df[(lstm_df['rmse'] >= lower_bound) & (lstm_df['rmse'] <= upper_bound)]


#IRM
q1 = irm_df['rmse'].quantile(0.25)
q3 = irm_df['rmse'].quantile(0.75)
iqr = q3 - q1

lower_bound = q1 - 1.5 * iqr
upper_bound = q3 + 1.5 * iqr

filtered_irm_df = irm_df[(irm_df['rmse'] >= lower_bound) & (irm_df['rmse'] <= upper_bound)]

# Merge on site
# merged_df = pd.merge(xgboost_df, lstm_df, irm_df, on='site', suffixes=('_xgb', '_lstm', '_irm'))

output_dir = "/cluster/project/math/akmete/MSc/XGBvsLSTM_OnSite/boxplotRMSE_xgblstmirm/"  # Specify the directory to save the plots

# Select relevant columns
# For LSTM, we skip 'test_loss' and use 'mse'
lstm_rmse = filtered_lstm_df['rmse']
# For XGBoost, we directly use 'rmse'
xgb_rmse = filtered_xgboost_df['rmse']
# For IRM, we directlu ise 'rmse'
irm_rmse = filtered_irm_df['rmse']

# Create a boxplot for the MSE comparison
plt.figure(figsize=(8, 6))
plt.boxplot(
    [xgb_rmse, lstm_rmse, irm_rmse],
    tick_labels=['XGBoost RMSE', 'LSTM RMSE', 'IRM RMSE']
)
# plt.title('Comparison of RMSE Distributions On Site Extrapolation in Time')
plt.ylabel('RMSE')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()

# Save the plot
output_path = os.path.join(output_dir, "rmse_comparison_boxplot.png")
plt.savefig(output_path, dpi=300)