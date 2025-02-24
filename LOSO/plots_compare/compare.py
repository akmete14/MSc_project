import pandas as pd

# Read csv
df_xgb = pd.read_csv('/cluster/project/math/akmete/MSc/LOSO/xgb.csv')
df_lstm = pd.read_csv('/cluster/project/math/akmete/MSc/LOSO/lstm.csv')

# Add suffix
df_xgb = df_xgb.add_suffix("_xgb")
df_lstm = df_lstm.add_suffix("_lstm")

# Remove suffix from 'site_left_out' column
df_xgb = df_xgb.rename(columns={"site_left_out_xgb": "site_left_out"})
df_lstm = df_lstm.rename(columns={"site_left_out_lstm": "site_left_out"})

# Merge both dataframes on 'site_left_out'
df_merged = pd.merge(df_xgb, df_lstm, on="site_left_out", how="outer")

import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# 1) Convert to a "long" format for easier plotting
df_melted = pd.melt(
    df_merged, 
    id_vars=["site_left_out"], 
    value_vars=["rmse_scaled_xgb", "rmse_scaled_lstm"],
    var_name="Model", 
    value_name="RMSE"
)

# 2) Create a box plot comparing RMSE distribution for both models
sns.boxplot(x="Model", y="RMSE", data=df_melted)
# plt.title("RMSE Distribution Across 290 Sites")
plt.savefig('XGBvsLSTM_RMSEdistribution.png')



### Another plot ###

# Step 1: Convert to long format
df_long = pd.melt(
    df_merged,
    id_vars=["site_left_out"],
    value_vars=["rmse_scaled_xgb", "rmse_scaled_lstm"],
    var_name="Model",
    value_name="RMSE"
)

# Step 2: Rename the model labels
df_long["Model"] = df_long["Model"].replace({
    "rmse_scaled_xgb": "XGBoost",
    "rmse_scaled_lstm": "LSTM"
})
print(df_long["Model"].unique())

# Step 3: Plot
plt.figure(figsize=(8, 6))
sns.histplot(
    data=df_long,
    x="RMSE",
    hue="Model",
    kde=True,
    alpha=0.5
)

plt.xlim(0, 0.05) 
plt.xticks(np.arange(0, 0.051, 0.01))
plt.ticklabel_format(style='plain', axis='x')
plt.xlabel("RMSE")
plt.ylabel("Count of Sites")
# plt.title("Histogram of RMSE")

# plt.legend(title="Model")  # Legend will have "XGBoost" and "LSTM"
plt.tight_layout()
plt.savefig("histogram_RMSE.png", dpi=300)