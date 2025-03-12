import os
import pandas as pd
import glob

# Get all CSV files matching the pattern
csv_files = glob.glob("onsite_*.csv")

# Read and concatenate all CSV files
df_list = [pd.read_csv(file) for file in csv_files]
merged_df = pd.concat(df_list, ignore_index=True)

# Sort by 'site' column
merged_df = merged_df.sort_values(by='site_id')

# Save to results.csv
merged_df.to_csv("results_corrected.csv", index=False)

print("Merged and sorted CSV saved as results.csv")
