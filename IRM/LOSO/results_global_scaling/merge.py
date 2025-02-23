import glob
import pandas as pd

# Find all CSV files that match the pattern
csv_files = glob.glob("fold_*_results.csv")

# Read each CSV into a DataFrame and combine them
df_list = [pd.read_csv(file) for file in csv_files]
merged_df = pd.concat(df_list, ignore_index=True)

# Sort by the 'site_left_out' column (change the column name if necessary)
merged_df = merged_df.sort_values(by='site_id')

# Save the merged and sorted DataFrame to a new CSV
merged_df.to_csv("results.csv", index=False)

print("Merged CSV saved as merged_results_LOSO.csv")
