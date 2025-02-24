import os
import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Define paths to CSV files
csv_files = {
    "LOGO": "/cluster/project/math/akmete/MSc/LOGO/LOGOCVXGB_all_metrics.csv",
    "LOGO + Weighted Groups": "/cluster/project/math/akmete/MSc/LOGO/LOGOCV_WeightedTraining_results.csv",
    "LOGO + Weighted Groups + Weighted Sites": "/cluster/project/math/akmete/MSc/LOGO_test/LOGOCV_WeightedTraining_weighted_within_group_results.csv"
}

# Define the correct column names mapping for the incorrect LOGO CSV
column_renaming = {
    "group_left_out": "GroupLeftOut",  # Example correction
    "mse": "MSE",
    "RMSE": "RMSE",
    "R2": "R2",
    "mae": "MAE"
}

# Read and combine all CSVs
df_list = []
for setting, path in csv_files.items():
    df = pd.read_csv(path)
    
    # Rename columns ONLY for the incorrect LOGO CSV
    if setting == "LOGO":
        df.rename(columns=column_renaming, inplace=True)
    
    df["Setting"] = setting  # Label the setting
    df_list.append(df)

# Combine all into a single DataFrame
df_all = pd.concat(df_list)

# Ensure the "plots" folder exists
plot_folder = "plots_comparison_balancedgrouping"
if not os.path.exists(plot_folder):
    os.makedirs(plot_folder)

# Define metrics to plot
metrics = ["MSE", "R2", "MAE", "RMSE"]

# Generate barplots for each metric with comparison
for metric in metrics:
    plt.figure(figsize=(12, 6))
    sns.barplot(data=df_all, x="GroupLeftOut", y=metric, hue="Setting")
    plt.xlabel("Group Left Out")
    plt.ylabel(metric)
    plt.xticks(rotation=45)  # Rotate x-labels if needed
    plt.legend(title="Setting")
    plt.tight_layout()
    plt.savefig(f"{plot_folder}/{metric}_comparison_barplot.png")
    plt.close()

# Boxplot for RMSE across settings
plt.figure(figsize=(6, 6))
sns.boxplot(data=df_all, x="Setting", y="RMSE")
plt.ylabel("RMSE")
plt.xlabel("Setting")
plt.tight_layout()
plt.savefig(f"{plot_folder}/RMSE_boxplot_comparison.png")
plt.close()

print(f"Plots saved in {plot_folder}/")

