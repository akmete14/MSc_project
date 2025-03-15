# Script to merge results when using SLURM parallelization
#Import libraries
import glob
import pandas as pd

# Find all CSV files with corresponding name
csv_files = glob.glob("results_site_*.csv")

# Read all csv to a dataframe
df_list = [pd.read_csv(file) for file in csv_files]
merged_df = pd.concat(df_list, ignore_index=True)

# Sort alphabetically 'site'
merged_df = merged_df.sort_values(by='site')

# Save the dataframe to one csv file
merged_df.to_csv("results.csv", index=False)

print("Merged CSV saved as merged_results_LOSO.csv")
