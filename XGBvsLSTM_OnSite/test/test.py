import pandas as pd
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

df_cleaned = df_filtered.dropna(subset=["rmse_xgb", "rmse_lstm", "rmse_irm"])

# Randomly select 20 unique sites
selected_sites = np.random.choice(df_cleaned["site"], size=20, replace=False)

# Filter dataframe to only include selected sites
df_selected = df_cleaned[df_cleaned["site"].isin(selected_sites)]

# Define x-axis positions for each model
models = ["XGB", "LSTM", "IRM"]
x_positions = [1, 2, 3]  # XGB at x=1, LSTM at x=2, IRM at x=3

plt.figure(figsize=(12, 6))

# Iterate through selected sites and plot each site's values
for i, site in enumerate(df_selected["site"]):
    values = [
        df_selected.loc[df_selected["site"] == site, "rmse_xgb"].values[0],
        df_selected.loc[df_selected["site"] == site, "rmse_lstm"].values[0],
        df_selected.loc[df_selected["site"] == site, "rmse_irm"].values[0]
    ]

    # Plot the dots
    plt.scatter(x_positions, values, label=site if i < 1 else "", s=60)

    # Connect the dots with a line
    plt.plot(x_positions, values, linestyle='-', marker='o', alpha=0.7)

# Customize the plot
plt.xticks(x_positions, models)  # Set x-axis labels
plt.xlabel("Model")
plt.ylabel("RMSE")
plt.title("RMSE On-Site Across Models")
plt.grid(True, linestyle="--", alpha=0.5)

# Show the plot
plt.tight_layout()
plt.savefig('test.png')