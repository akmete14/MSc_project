import pandas as pd
import numpy as np

# Read the dataframe
df = pd.read_csv('/cluster/project/math/akmete/MSc/preprocessing/df_balanced_groups.csv')
print(df.columns)

# Get rid of cluster and Unnamed: 0 columns
df = df.drop(columns=['cluster', 'Unnamed: 0'])
print(df.columns)

# Now, we do random grouping of the sites
# Get unique site IDs
unique_sites = df['site_id'].unique()

# Randomly assign each site ID to one of 10 clusters
np.random.seed(42)  # Set seed for reproducibility
cluster_assignments = {site: cluster for site, cluster in zip(unique_sites, np.random.randint(1, 11, size=len(unique_sites)))}

# Map the cluster assignments back to the dataframe
df['cluster'] = df['site_id'].map(cluster_assignments)

# Verify the result
print(df.head())
print(len(df))

# Number of rows of data in each cluster
rows_per_cluster = df['cluster'].value_counts().sort_index()
print("Number of rows per cluster:")
print(rows_per_cluster)

# Number of unique sites in each cluster
sites_per_cluster = df.groupby('cluster')['site_id'].nunique()
print("\nNumber of unique sites per cluster:")
print(sites_per_cluster)


df.to_csv('df_random_grouping.csv')

