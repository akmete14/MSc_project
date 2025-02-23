import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Load the CSV file
df = pd.read_csv("/cluster/project/math/akmete/MSc/IRM/OnSite/results.csv")

# Create the boxplot
plt.figure(figsize=(8, 6))
plt.boxplot(df["rmse"], labels=["RMSE"])

# Use a log scale for better visualization
plt.yscale("log")
plt.ylabel("Error Value (log scale)")
plt.title("Boxplot of RMSE (Log Scale)")

# Save the figure
plt.savefig("boxplot_rmse_log.png", dpi=300, bbox_inches='tight')
plt.close()
