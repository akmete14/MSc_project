import pandas as pd
import glob
import numpy as np

# Load the existing results.csv
results_df = pd.read_csv("results.csv")

# Get the new CSV files that were forgotten
new_csv_files = glob.glob("results_site_*.csv")

# Read and concatenate the new CSV files
new_df_list = [pd.read_csv(file) for file in new_csv_files]
new_data_df = pd.concat(new_df_list, ignore_index=True)
new_data_df["ensemble_rmse"] = np.sqrt(new_data_df["ensemble_mse_scaled"])
new_data_df["full_rmse"] = np.sqrt(new_data_df["full_mse_scaled"])

# Merge with the existing results
updated_df = pd.concat([results_df, new_data_df], ignore_index=True)

# Sort by 'site' column
updated_df = updated_df.sort_values(by='site')

# Save the updated results.csv
updated_df.to_csv("results_2.csv", index=False)

print("Updated results.csv with new data and sorted by site.")

