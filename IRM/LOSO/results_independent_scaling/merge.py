import pandas as pd
import glob
import os

# Define the directory where CSV files are stored
file_directory = "/cluster/project/math/akmete/MSc/IRM/LOSO/results_independent_scaling/"

# Create the pattern to match only files named as "fold_i_results.csv"
file_pattern = os.path.join(file_directory, "fold_*_results.csv")

# Get list of matching CSV files
csv_files = sorted(glob.glob(file_pattern))

# Filter files to include only up to fold_90_results.csv
csv_files = [f for f in csv_files if int(os.path.basename(f).split("_")[1]) <= 289]

# Check if there are matching files
if not csv_files:
    print("No matching CSV files found.")
    exit()

# Initialize an empty list to store DataFrames
dfs = []

# Read each CSV and append it to the list
for file in csv_files:
    df = pd.read_csv(file)
    dfs.append(df)

# Concatenate all DataFrames
merged_df = pd.concat(dfs, ignore_index=True)
merged_df = merged_df.sort_values(by="site_id")

# Save to a new CSV file
output_path = os.path.join(file_directory, "results.csv")
merged_df.to_csv(output_path, index=False)

print(f"Merged CSV saved to: {output_path}")