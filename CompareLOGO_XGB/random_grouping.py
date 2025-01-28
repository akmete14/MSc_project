import os
import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

df = pd.read_csv('/cluster/project/math/akmete/MSc/LOGO/XGB_random_grouping_all_metrics.csv')

# 1) Ensure the "plots" folder exists
if not os.path.exists("plots_random_grouping"):
    os.makedirs("plots_random_grouping")

# 2) List of metrics you want to barplot
metrics = ["mse", "R2", "Relative Error", "MAE", "RMSE"]

# 3) Barplot for each metric
for metric in metrics:
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df, x="group_left_out", y=metric, color="skyblue")
    plt.title(f"Barplot of {metric}")
    plt.xlabel("Group Left Out")
    plt.ylabel(metric)
    plt.xticks(rotation=45)    # Rotate x-labels if many groups
    plt.tight_layout()
    plt.savefig(f"plots_random_grouping/{metric}_barplot.png")
    plt.close()  # Close the figure so it doesn't show

# 4) Boxplot for RMSE
plt.figure(figsize=(6, 6))
sns.boxplot(data=df, y="RMSE", color="lightgreen")
plt.title("Boxplot of RMSE across groups")
plt.ylabel("RMSE")
plt.tight_layout()
plt.savefig("plots_random_grouping/RMSE_boxplot.png")
plt.close()
