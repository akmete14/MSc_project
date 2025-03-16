# Merging csvs
# Import libraries
import glob
import pandas as pd

# Find all CSVs
csv_files = glob.glob("results_site_*.csv")

# Read in each csv and merge to one dataframe
df_list = [pd.read_csv(file) for file in csv_files]
merged_df = pd.concat(df_list, ignore_index=True)

# sort by test site
merged_df = merged_df.sort_values(by='site')

# save merged dataframe as csv
merged_df.to_csv("results_lasso_modified.csv", index=False)

print("Merged CSV saved as merged_results_LOSO.csv")


