# Import libraries
import glob
import pandas as pd

# Find all CSV files with matching name
csv_files = glob.glob("results_site_*.csv")

# Read each CSV and concatenate to one dataframe
df_list = [pd.read_csv(file) for file in csv_files]
merged_df = pd.concat(df_list, ignore_index=True)

# Sort by test_site
merged_df = merged_df.sort_values(by='test_site')

# Save merged dataframe to a CSV
merged_df.to_csv("results_50sites.csv", index=False)

print("Merged CSV saved as merged_results_LOSO.csv")