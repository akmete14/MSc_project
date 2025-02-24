import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd


df = pd.read_csv('/cluster/project/math/akmete/MSc/LOSO/xgb_independent_scaling.csv')
dg = pd.read_csv('/cluster/project/math/akmete/MSc/LOSO/xgb.csv')

# Assuming df and dg each have columns 'site' and 'rmse'
# Extract the RMSE values from each DataFrame
rmse_df = df['rmse_scaled']
rmse_dg = dg['rmse_scaled']

plt.figure(figsize=(8, 6))
plt.boxplot([rmse_df, rmse_dg], labels=['Independent Scaling', 'Global Scaling'])
plt.title('Comparison of RMSE Distributions')
plt.ylabel('RMSE')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('rmse_boxplot_global_vs_independent_scaling.png')
