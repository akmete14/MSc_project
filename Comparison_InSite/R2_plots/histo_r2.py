import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load datasets
df_xgb = pd.read_csv('/cluster/project/math/akmete/MSc/XGBoost/InSite/results_2.csv')
df_lstm = pd.read_csv('/cluster/project/math/akmete/MSc/LSTM/InSite/results_modified.csv')
df_irm = pd.read_csv('/cluster/project/math/akmete/MSc/IRM/OnSite/results_corrected.csv')
df_lr = pd.read_csv('/cluster/project/math/akmete/MSc/LR/In_Site/results_2.csv')


# Standardize column names
df_xgb.rename(columns={'r2_score': 'r2'}, inplace=True)
df_irm.rename(columns={'site_id': 'site'}, inplace=True)
df_lr.rename(columns={'site_left_out': 'site'}, inplace=True)
df_lr.rename(columns={'r2_score': 'r2'}, inplace=True)

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

# Function to remove upper outliers using IQR instead of 95th percentile
def remove_upper_outliers(df, column='r2'):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    upper_bound = Q3 + 1.5 * IQR  # Standard IQR rule
    return df[df[column] <= upper_bound]

# Apply outlier removal
df_lr_filtered = remove_upper_outliers(df_lr)
df_irm_filtered = remove_upper_outliers(df_irm)
df_xgb_filtered = remove_upper_outliers(df_xgb)
df_lstm_filtered = remove_upper_outliers(df_lstm)

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
