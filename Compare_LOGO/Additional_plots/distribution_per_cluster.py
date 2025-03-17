# Import libraries
import matplotlib.pyplot as plt
import pandas as pd

# Read data
df = pd.read_csv('/cluster/project/math/akmete/MSc/preprocessing/df_balanced_groups_onevegindex.csv')

# Set frame of plot
plt.figure(figsize=(8,6))

# Define data to plot
ax = df.boxplot(column="GPP", by="cluster", showfliers=False, vert=False)

# No title
ax.set_title("")

# Remove the default suptitle
plt.suptitle("")

# Only keep axes labels
plt.xlabel("GPP")
plt.ylabel("Cluster")

plt.savefig('target_boxplot_per_cluster.png')