import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# Load CSV files
xgboost_df = pd.read_csv('XGB_baseline_single_site.csv')
lstm_df = pd.read_csv('LSTM_baseline_single_site.csv')

# Merge on site
merged_df = pd.merge(xgboost_df, lstm_df, on='site', suffixes=('_xgb', '_lstm'))

output_dir = "/cluster/project/math/akmete/MSc/XGBvsLSTM_OnSite/plots_base/"  # Specify the directory to save the plots

# Criterion 1: Select 20 sites with the largest absolute difference in MSE
merged_df['mse_diff'] = abs(merged_df['mse_xgb'] - merged_df['mse_lstm'])
top_20_sites = merged_df.nlargest(20, 'mse_diff')

# Criterion 2: Randomly select 20 sites
random_20_sites = merged_df.sample(20, random_state=42)

# Choose which subset to visualize
#subset = top_20_sites  # or random_20_sites
subset = random_20_sites


metrics = ['mse', 'r2', 'relative_error', 'mae', 'rmse']

for metric in metrics:
    plt.figure(figsize=(12, 6))
    x = range(len(subset['site']))
    plt.bar(x, subset[f'{metric}_xgb'], width=0.4, label='XGBoost', align='center')
    plt.bar([p + 0.4 for p in x], subset[f'{metric}_lstm'], width=0.4, label='LSTM', align='center')
    plt.xticks([p + 0.2 for p in x], subset['site'], rotation=90)
    plt.ylabel(metric)
    # plt.title(f'Comparison of {metric} for Selected Sites')
    plt.legend()
    plt.tight_layout()
    # Save the plot
    plt.savefig(f"{output_dir}{metric}_comparison_barplot_random_20_sites.png", dpi=300)
    plt.close()  # Close the plot to free memory

# Select relevant columns
# For LSTM, we skip 'test_loss' and use 'mse'
lstm_rmse = lstm_df['rmse']
# For XGBoost, we directly use 'mse'
xgb_rmse = xgboost_df['rmse']

# Create a boxplot for the MSE comparison
plt.figure(figsize=(8, 6))
plt.boxplot(
    [xgb_rmse, lstm_rmse],
    tick_labels=['XGBoost RMSE', 'LSTM RMSE']
)
# plt.title('Comparison of RMSE Distributions On Site Extrapolation in Time')
plt.ylabel('RMSE')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()

# Save the plot
plt.savefig("rmse_comparison_boxplot.png", dpi=300)