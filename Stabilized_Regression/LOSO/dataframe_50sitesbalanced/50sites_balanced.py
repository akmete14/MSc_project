import pandas as pd
import numpy as np

# Sample DataFrame (assuming you already have one)
df = pd.read_csv("/cluster/project/math/akmete/MSc/preprocessing/df_balanced_groups_onevegindex.csv")

# Ensure site_id is categorical to prevent issues
df['site_id'] = df['site_id'].astype(str)
df['cluster'] = df['cluster'].astype(int)  # Ensure cluster is integer

# Create an empty list to store selected site_ids
selected_sites = []

# Loop over each cluster and randomly select 5 unique site_ids
for cluster_id in range(10):  # Clusters 0 to 9
    cluster_sites = df[df['cluster'] == cluster_id]['site_id'].unique()  # Get unique site_ids in this cluster
    
    # Ensure there are at least 5 sites in the cluster
    if len(cluster_sites) >= 5:
        selected = np.random.choice(cluster_sites, 5, replace=False)  # Randomly pick 5 sites
    else:
        selected = cluster_sites  # If less than 5 sites, take all available

    selected_sites.extend(selected)  # Store selected site_ids

# Filter the original DataFrame to only include selected sites
df_selected = df[df['site_id'].isin(selected_sites)]

df_selected.to_csv("50sites_balanced.csv", index=False)
