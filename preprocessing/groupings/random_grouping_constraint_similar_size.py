import pandas as pd
import numpy as np

# Read the dataframe
df = pd.read_csv('/cluster/project/math/akmete/MSc/preprocessing/df_balanced_groups.csv')

# Remove unnecessary columns
df = df.drop(columns=['cluster', 'Unnamed: 0'])

# Get unique sites and the row count for each site
site_row_counts = df['site_id'].value_counts()

# Total number of rows and approximate target rows per cluster
total_rows = len(df)
num_clusters = 10
target_rows_per_cluster = total_rows // num_clusters

# Randomly shuffle the sites
np.random.seed(42)  # For reproducibility
shuffled_sites = site_row_counts.index.to_numpy()
np.random.shuffle(shuffled_sites)

# Assign clusters while balancing row counts
clusters = {}
cluster_sizes = {i: 0 for i in range(1, num_clusters + 1)}

for site in shuffled_sites:
    # Find the cluster with the smallest size that doesn't exceed the target
    assigned_cluster = min(
        cluster_sizes,
        key=lambda x: cluster_sizes[x] if cluster_sizes[x] < target_rows_per_cluster else float('inf')
    )
    clusters[site] = assigned_cluster
    cluster_sizes[assigned_cluster] += site_row_counts[site]

# Map the cluster assignments back to the dataframe
df['cluster'] = df['site_id'].map(clusters)

# Verify the row counts per cluster
rows_per_cluster = df['cluster'].value_counts().sort_index()
print("Rows per cluster:")
print(rows_per_cluster)

# Save the results if needed
df.to_csv('df_random_grouping_constraint_similar_size.csv', index=False)
