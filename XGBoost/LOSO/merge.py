# This file merges all obtained csvs when running the LOSO experiment parallel
import glob
import pandas as pd

# Find all CSV files matching the name
csv_files = glob.glob("results_LOSO_*.csv")

# Read each CSV into a DataFrame and combine them
df_list = [pd.read_csv(file) for file in csv_files]
merged_df = pd.concat(df_list, ignore_index=True)

# Sort by the 'site_left_out' column
merged_df = merged_df.sort_values(by='site')

# Save merged csv
merged_df.to_csv("results.csv", index=False)

print("Merged CSV saved as merged_results_LOSO.csv")