# Import libraries
import glob
import pandas as pd

# Find all CSV files
csv_files = glob.glob("results_site_*.csv")

# Read each CSV into a dataframe and concatenate
df_list = [pd.read_csv(file) for file in csv_files]
merged_df = pd.concat(df_list, ignore_index=True)

# Sort by sitename
merged_df = merged_df.sort_values(by='site')

# Save the merged dataframe to csv
merged_df.to_csv("results_lasso_modified.csv", index=False)

print("Merged CSV saved as merged_results_LOSO.csv")
