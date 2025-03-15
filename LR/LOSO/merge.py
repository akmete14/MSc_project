# Import libraries
import glob
import pandas as pd

# Find all csv files matching the name
csv_files = glob.glob("results_LOSO_*.csv")

# Read all csv files and combine them to one dataframe
df_list = [pd.read_csv(file) for file in csv_files]
merged_df = pd.concat(df_list, ignore_index=True)

# Sort by the dataframe by 'site_left_out' column
merged_df = merged_df.sort_values(by='site_left_out')

# Save the merged dataframe to a csv
merged_df.to_csv("results_2.csv", index=False)

print("Merged CSV saved as merged_results_LOSO.csv")