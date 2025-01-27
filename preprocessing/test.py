import pandas as pd

df = pd.read_csv('grouping_equal_size(1).csv')
df = df.drop(columns=['Unnamed: 0'])
print(df['balanced_cluster'].value_counts())
print(df.head())

import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend
import matplotlib.pyplot as plt

# If your 'balanced_cluster' column is numeric (e.g. 0..9, or 1..10), this will work fine.
# If it's textual, we can still iterate over unique group labels; the code below handles both.

plt.figure(figsize=(10, 6))

# Get the unique group labels:
group_labels = df['balanced_cluster'].unique()

# Create a discrete color map with one color per group (tab10 has up to 10 distinct colors).
# If you have more than 10 groups, consider using another colormap like 'tab20'.
cmap = plt.cm.get_cmap('tab10', len(group_labels))

for idx, group_value in enumerate(group_labels):
    subset = df[df['balanced_cluster'] == group_value]
    
    plt.scatter(
        subset['longitude'],
        subset['latitude'],
        color=cmap(idx),
        label=f'Group {group_value}',
        alpha=0.8
    )


plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.legend()
plt.tight_layout()
plt.savefig('test.png')
plt.close()
