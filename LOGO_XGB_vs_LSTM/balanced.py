import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os


df_xgb = pd.read_csv('/cluster/project/math/akmete/MSc/LOGO/LOGOCVXGB_all_metrics.csv')
df_lstm = pd.read_csv('/cluster/project/math/akmete/MSc/LSTM/LOGOCV_balanced_grouping.csv')

# Rename df_xgb so that both df have same naming
df_xgb = df_xgb.rename(columns={'R2': 'r2', 'Relative Error': 'relative_error','MAE': 'mae', 'RMSE': 'rmse'})

# Create directory if it doesn't exist
output_dir = "plots_balanced"
os.makedirs(output_dir, exist_ok=True)

# List of metrics to plot
metrics = ['mse', 'r2', 'relative_error', 'mae', 'rmse']

# Iterate over metrics
for metric in metrics:
    plt.figure(figsize=(10, 6))

    # Extract data
    x = df_xgb['group_left_out']
    width = 0.4  # Width of bars

    # Sort values to ensure alignment
    df_xgb_sorted = df_xgb.sort_values('group_left_out')
    df_lstm_sorted = df_lstm.sort_values('group_left_out')

    # Bar positions
    x_positions = np.arange(len(x))

    # Plot bars for XGBoost and LSTM
    plt.bar(x_positions - width/2, df_xgb_sorted[metric], width=width, label='XGBoost', alpha=0.7)
    plt.bar(x_positions + width/2, df_lstm_sorted[metric], width=width, label='LSTM', alpha=0.7)

    # Labels and title
    plt.xticks(x_positions, df_xgb_sorted['group_left_out'], rotation=45)
    plt.ylabel(metric)
    plt.title(f'Comparison of {metric} by Group Left Out')
    plt.legend()

    # Save plot as PNG
    plot_path = os.path.join(output_dir, f"{metric}.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()  # Close the figure to free memory

print(f"Plots saved in '{output_dir}' folder.")