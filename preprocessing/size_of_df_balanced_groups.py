import pandas as pd

df = pd.read_csv('/cluster/project/math/akmete/MSc/preprocessing/df_balanced_groups.csv')

cluster_counts = df['cluster'].value_counts()
print(cluster_counts)