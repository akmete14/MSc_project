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
df_lr = df_lr[['Model', 'mae']]
df_xgb = df_xgb[['Model', 'mae']]
df_lstm = df_lstm[['Model', 'mae']]
df_irm = df_irm[['Model', 'mae']]

# Function to remove upper outliers using IQR instead of 95th percentile
def remove_upper_outliers(df, column='mae'):
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


# ===============================
# Plot Histogram Comparing Models (Without Upper Outliers)
# ===============================
plt.figure(figsize=(8, 6))

# Define x-axis range
min_mae = df_combined['mae'].min()
x_max_limit = 0.06  # Hard limit for the right side

# Plot histogram
sns.histplot(
    data=df_combined,
    x="mae",  
    hue="Model",  
    bins=40,
    alpha=0.3  # Slight transparency
)

# Manually add KDE plots with x-axis clip
for model in df_combined["Model"].unique():
    subset = df_combined[df_combined["Model"] == model]
    sns.kdeplot(
        subset["mae"],
        label=f"{model} KDE",
        clip=(min_mae, x_max_limit),  # KDE limited to x_max_limit
        linewidth=1.5
    )

# Set x-axis limits explicitly
plt.xlim(min_mae, x_max_limit)

# Formatting axes
plt.xticks(np.linspace(min_mae, x_max_limit, 8))
plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.3f}'))  # More precision for small values

# Labels
plt.xlabel("MAE")
plt.ylabel("Count of Sites")
#plt.title("Histogram of MAE Across Models (Clipped at 0.05)")

# Layout adjustments
plt.tight_layout()

# Save the plot
plt.savefig("histogram_mae.png", dpi=300)