import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Read in the data
df_xgb = pd.read_csv('/cluster/project/math/akmete/MSc/XGBoost/LOSO/results.csv')
df_lstm = pd.read_csv('/cluster/project/math/akmete/MSc/LSTM/LOSO/results.csv')
df_irm = pd.read_csv('/cluster/project/math/akmete/MSc/IRM/LOSO/results_global_scaling/results.csv')
df_lr = pd.read_csv('/cluster/project/math/akmete/MSc/LR/LOSO/results_2.csv')

# Standardize column names
df_xgb.rename(columns={'r2_score': 'r2'}, inplace=True)
df_irm.rename(columns={'site_id': 'site'}, inplace=True)
df_lr.rename(columns={'site_left_out': 'site'}, inplace=True)

# Add model identifiers
df_xgb['Model'] = 'XGBoost'
df_lstm['Model'] = 'LSTM'
df_irm['Model'] = 'IRM'
df_lr['Model'] = 'LR'

# Select relevant columns
df_lr = df_lr[['Model', 'r2']]
df_xgb = df_xgb[['Model', 'r2']]
df_lstm = df_lstm[['Model', 'r2']]
df_irm = df_irm[['Model', 'r2']]



# Combine all models into a single DataFrame
df_combined = pd.concat([df_lr, df_xgb, df_lstm, df_irm], ignore_index=True)

# Remove negative R² values (if applicable)
df_combined = df_combined[df_combined['r2'] >= -0.5]


# ===============================
# Plot Histogram Comparing Models (Without Upper Outliers)
# ===============================
plt.figure(figsize=(8, 6))

sns.histplot(
    data=df_combined,
    x="r2",  
    hue="Model",  
    kde=True,  
    alpha=0.4,  # More transparent
    bins=40
)

# Adjust x-axis dynamically based on actual r2 values
plt.xlim(df_combined['r2'].min(), df_combined['r2'].max())

# Formatting axes
plt.xticks(np.linspace(df_combined['r2'].min(), df_combined['r2'].max(), 10))
plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.2f}'))

# Labels
plt.xlabel("R²")
plt.ylabel("Count of Sites")

# Layout adjustments
plt.tight_layout()

# Save the plot
plt.savefig("histogram_R2.png", dpi=300)
