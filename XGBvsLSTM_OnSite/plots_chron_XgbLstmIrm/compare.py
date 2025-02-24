import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Read CSV files
df_xgb = pd.read_csv('/cluster/project/math/akmete/MSc/XGBvsLSTM_OnSite/xgb_extrapolation_in_time_single_site.csv')
df_lstm = pd.read_csv('/cluster/project/math/akmete/MSc/XGBvsLSTM_OnSite/LSTM_extrapolation_in_time_single_site.csv')
df_irm = pd.read_csv('/cluster/project/math/akmete/MSc/IRM/OnSite/results.csv')

# Add suffix
df_xgb = df_xgb.add_suffix("_xgb")
df_lstm = df_lstm.add_suffix("_lstm")
df_irm = df_irm.add_suffix("_irm")

# Remove suffix from 'site' column
df_xgb = df_xgb.rename(columns={"site_xgb": "site"})
df_lstm = df_lstm.rename(columns={"site_lstm": "site"})
df_irm = df_irm.rename(columns={"site_irm": "site"})

# Merge dataframes on 'site'
df_merged = pd.merge(df_xgb, df_lstm, on="site", how="outer")
df_merged = pd.merge(df_merged, df_irm, on="site", how="outer")

# List of RMSE columns to process
rmse_columns = ["rmse_xgb", "rmse_lstm", "rmse_irm"]

# Function to remove outliers using IQR method
def remove_outliers(df, columns):
    for col in columns:
        Q1 = df[col].quantile(0.25)  # First quartile (25th percentile)
        Q3 = df[col].quantile(0.75)  # Third quartile (75th percentile)
        IQR = Q3 - Q1  # Interquartile range
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
    return df

# Remove outliers
df_filtered = remove_outliers(df_merged, rmse_columns)

# Convert to long format for plotting
df_melted = pd.melt(
    df_filtered, 
    id_vars=["site"], 
    value_vars=rmse_columns,
    var_name="Model", 
    value_name="RMSE"
)

# Rename models for clarity
df_melted["Model"] = df_melted["Model"].replace({
    "rmse_xgb": "XGBoost",
    "rmse_lstm": "LSTM",
    "rmse_irm": "IRM"
})

# Box plot (without outliers)
plt.figure(figsize=(8, 6))
sns.boxplot(x="Model", y="RMSE", data=df_melted)
plt.xlabel("Model")
plt.ylabel("RMSE")
#plt.title("On Site Extrapolation RMSE Distribution (Outliers Removed)")
plt.savefig("XGB_LSTM_IRM_RMSEdistribution_filtered.png", dpi=300)

# Histogram with KDE (without outliers)
plt.figure(figsize=(8, 6))
sns.histplot(
    data=df_melted,
    x="RMSE",
    hue="Model",
    kde=True,
    alpha=0.5
)
plt.xlim(0, 0.10)
plt.xticks(np.arange(0, 0.10, 0.01))
plt.ticklabel_format(style='plain', axis='x')
plt.xlabel("RMSE")
plt.ylabel("Count of Sites")
#plt.title("On Site Extrapolation Histogram of RMSE (Outliers Removed)")
plt.tight_layout()
plt.savefig("histogram_RMSE_filtered.png", dpi=300)
