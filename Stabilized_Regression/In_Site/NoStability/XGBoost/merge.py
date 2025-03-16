# This script can be used to merge the single csv files obtained from insite_screened.py
# Import libraries
import glob
import pandas as pd

# Find all csv files matching the name
csv_files = glob.glob("results_site_*.csv")

# Read all csv in and combine them to one dataframe
df_list = [pd.read_csv(file) for file in csv_files]
merged_df = pd.concat(df_list, ignore_index=True)

# Sort dataframe by site column
merged_df = merged_df.sort_values(by='site')

# Save merged dataframe to csv
merged_df.to_csv("results_screened_O_hat_modified_top7.csv", index=False)

print("Merged CSV saved as merged_results_LOSO.csv")
