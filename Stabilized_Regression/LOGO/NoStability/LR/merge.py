# Import libraries
import glob
import pandas as pd

# Find all CSV files
csv_files = glob.glob("results_cluster_*.csv")

# Read each CSV and concatenate to one dataframe
df_list = [pd.read_csv(file) for file in csv_files]
merged_df = pd.concat(df_list, ignore_index=True)

# Sort dataframe by test_cluster
merged_df = merged_df.sort_values(by='test_cluster')

# Save the merged dataframe as CSV
merged_df.to_csv("results_lasso_withOhat.csv", index=False)

print("Merged CSV saved as merged_results_LOSO.csv")