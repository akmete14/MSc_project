# This script is only needed when parallelizing the job using SLURM.
# Because then, this script merges all individual csv we got from running the job
import glob
import pandas as pd

# Find all CSV files matching the name
csv_files = glob.glob("results_site_*.csv")

# Read each CSV into a DataFrame and combine them
df_list = [pd.read_csv(file) for file in csv_files]
merged_df = pd.concat(df_list, ignore_index=True)

# Sort by the 'site_left_out' column
merged_df = merged_df.sort_values(by='site')

# Save the obtained dataframe (we might now delete the individual csvs)
merged_df.to_csv("results_2.csv", index=False)

print("Merged CSV saved as merged_results_InSite.csv")