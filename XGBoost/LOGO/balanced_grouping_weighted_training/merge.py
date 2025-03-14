# Use this script to merge the individual csv
import glob
import pandas as pd

# Find all CSV files that the name
csv_files = glob.glob("results_LOGO_*.csv")

# Read each CSV into a DataFrame and combine them
df_list = [pd.read_csv(file) for file in csv_files]
merged_df = pd.concat(df_list, ignore_index=True)

# Sort by the 'cluster' column
merged_df = merged_df.sort_values(by='cluster')

# Save merged dataframe to one resulting csv
merged_df.to_csv("results.csv", index=False)

print("Merged CSV saved")