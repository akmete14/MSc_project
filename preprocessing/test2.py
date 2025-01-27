import pandas as pd

df = pd.read_csv('/cluster/project/math/akmete/MSc/preprocessing/df_balanced_groups.csv')
df = df.drop(columns=['Unnamed: 0'])

# Step 1: Group by 'site_id' and take the first assigned group (or deduplicate)
unique_sites = df.groupby('site_id')['cluster'].first().reset_index()

# Step 2: Count the number of unique sites in each group
group_counts = unique_sites['cluster'].value_counts()

# Step 3: (Optional) Sort or transform results into a DataFrame
group_counts_df = group_counts.reset_index()
group_counts_df.columns = ['cluster', 'count']

# Display the result
print(group_counts_df)
