# Import libraries
import os
import pandas as pd
import glob

# Get all csv files with matching file name
csv_files = glob.glob("onsite_*.csv")

# Read all csv and concatenate them
df_list = [pd.read_csv(file) for file in csv_files]
merged_df = pd.concat(df_list, ignore_index=True)

# Sort datafrmae by 'site_id' column
merged_df = merged_df.sort_values(by='site_id')

# Save to one csv
merged_df.to_csv("results_corrected.csv", index=False)

print("Merged and sorted CSV saved as results.csv")
