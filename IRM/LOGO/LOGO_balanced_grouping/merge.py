# Import libraries
import glob
import pandas as pd

# Find all csv files with matching name
csv_files = glob.glob("cluster_*_results.csv")

# Read all csv files and concatenate them to one dataframe
df_list = [pd.read_csv(file) for file in csv_files]
merged_df = pd.concat(df_list, ignore_index=True)

# Sort by cluster
merged_df = merged_df.sort_values(by='cluster')

# Save merged dataframe to new csv
merged_df.to_csv("results.csv", index=False)

print("Merged CSV's")
