import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Load the data from the CSV
df = pd.read_csv("/cluster/project/math/akmete/MSc/LOGO/LOGOCVXGB_all_metrics.csv")

# Set Seaborn style for better visuals
sns.set(style="whitegrid")

# Define the folder to save plots
plot_folder = "plot_metrics"

# Ensure the folder exists
if not os.path.exists(plot_folder):
    os.makedirs(plot_folder)

# Barplots for all metrics
metrics = ["mse", "R2", "Relative Error", "MAE", "RMSE"]

for metric in metrics:
    plt.figure(figsize=(10, 6))
    sns.barplot(x="group_left_out", y=metric, data=df, palette="viridis")
    plt.title(f"Bar Plot of {metric} by Group")
    plt.xlabel("Group Left Out")
    plt.ylabel(metric)
    plt.tight_layout()
    
    # Save each barplot in the specified folder
    file_path = os.path.join(plot_folder, f"barplot_{metric}.png")
    plt.savefig(file_path, dpi=300)
    plt.close()

# Boxplot for RMSE
plt.figure(figsize=(10, 6))
sns.boxplot(y="RMSE", data=df, palette="coolwarm")
plt.title("Box Plot of RMSE")
plt.ylabel("RMSE")
plt.tight_layout()

# Save the boxplot in the specified folder
file_path = os.path.join(plot_folder, "boxplot_RMSE.png")
plt.savefig(file_path, dpi=300)
plt.close()

print(f"Plots saved successfully in the folder: {plot_folder}")
