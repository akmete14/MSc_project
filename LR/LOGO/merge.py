# When running logo.py in parallel using SLURM, then use this script to merge all csvs
# Import libraries
import glob
import pandas as pd

# Find all csv files with the corresponding name
csv_files = glob.glob("results_LOGO_*.csv")

# Read all csv files and merge them to one dataframe
df_list = [pd.read_csv(file) for file in csv_files]
merged_df = pd.concat(df_list, ignore_index=True)

# Sort by the results by cluster number
merged_df = merged_df.sort_values(by='cluster_left_out')

# Save the dataframe to a csv
merged_df.to_csv("results_2.csv", index=False)

print("Merged CSV saved as merged_results_LOSO.csv")