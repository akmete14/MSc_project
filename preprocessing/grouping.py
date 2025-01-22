import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import xarray as xr
from tqdm import tqdm
from pyclustering.cluster.kmedoids import kmedoids
from pyclustering.utils import calculate_distance_matrix

path = '/cluster/project/math/akmete/MSc/Data/'
files = [f for f in os.listdir(path) if f.endswith('.nc')]
files.sort()
assert len(files) % 3 == 0
files = {files[0 + 3 * i][:6]: (files[0 + 3 * i], files[1 + 3 * i], files[2 + 3 * i]) for i in range(len(files) // 3)}
sites = list(files.keys())

def initialize_dataframe(file1, file2, file3, path, sample_fraction=0.25):
    # Open data
    ds = xr.open_dataset(path + file1, engine='netcdf4')
    ds = ds[['longitude', 'latitude']]

    # Convert to DataFrame
    df = ds.to_dataframe().reset_index()
    df = df.drop(columns=['x', 'y'])

    return df

coordinates = []
for site in tqdm(sites):
    df = initialize_dataframe(*files[site], path=path)
    longitude = df['longitude'].iloc[0]
    latitude = df['latitude'].iloc[0]
    sitename = site
    coordinates.append({'site': sitename, 'longitude': longitude, 'latitude': latitude})

df_coordinates = pd.DataFrame(coordinates)

# Decide on the number of clusters
n_clusters = 10

# Extract coordinates as a numpy array
X = df_coordinates[['latitude', 'longitude']].to_numpy()

# Calculate distance matrix
distance_matrix = calculate_distance_matrix(X)

# Initialize medoids (randomly selecting n_clusters points as medoids)
initial_medoids = np.random.choice(range(len(X)), n_clusters, replace=False).tolist()

# Perform K-Medoids clustering
kmedoids_instance = kmedoids(distance_matrix, initial_medoids, data_type='distance_matrix')
kmedoids_instance.process()
clusters = kmedoids_instance.get_clusters()

# Assign cluster labels to the DataFrame
cluster_labels = [-1] * len(X)
for cluster_id, cluster_points in enumerate(clusters):
    for point_index in cluster_points:
        cluster_labels[point_index] = cluster_id

df_coordinates['group'] = cluster_labels

# Visualize grouping
plt.figure(figsize=(8, 6))
for cluster_id in df_coordinates['group'].unique():
    cluster_points = df_coordinates[df_coordinates['group'] == cluster_id]
    plt.scatter(cluster_points['longitude'], cluster_points['latitude'], label=f"Group {cluster_id}")

plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.title(f"KMedoids Clustering (n={n_clusters})")
plt.legend()
plt.savefig('grouping_equal_size.png')
plt.close()
