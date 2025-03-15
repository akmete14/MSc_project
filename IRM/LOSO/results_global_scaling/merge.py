# Impot libraries
import glob
import pandas as pd

# Find all csv files with matching name
csv_files = glob.glob("fold_*_results.csv")

# Read in all csv and concatenate to one dataframe
df_list = [pd.read_csv(file) for file in csv_files]
merged_df = pd.concat(df_list, ignore_index=True)

# Sort the dataframe by site_id column
merged_df = merged_df.sort_values(by='site_id')

# Save the merged df to a csv
merged_df.to_csv("results.csv", index=False)

print("Merged CSV saved as merged_results_LOSO.csv")
