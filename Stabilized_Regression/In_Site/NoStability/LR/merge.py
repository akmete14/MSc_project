import pandas as pd
import glob
import os

# Define the path where CSV files are located
csv_folder = "/cluster/project/math/akmete/MSc/Stabilized_Regression/In_Site/NoStability/LR/"

# Find all files matching the pattern "results_site_*"
csv_files = glob.glob(os.path.join(csv_folder, "**/results_site_*.csv"), recursive=True)

# Use a temporary location for writing first (e.g., /tmp/)
temp_merged_csv = "/tmp/merged_results.csv"
final_merged_csv = os.path.join(csv_folder, "merged_results.csv")

# Open the merged CSV file in append mode
header_written = False  # Ensure header is written only once

with open(temp_merged_csv, "w") as merged_file:
    for file in csv_files:
        try:
            # Read the CSV file (each file has only one row)
            df = pd.read_csv(file)

            # Append to the merged file
            df.to_csv(merged_file, index=False, header=not header_written, mode="a")
            header_written = True  # Ensure header is written only once

            # Delete the CSV immediately after processing (free space)
            os.remove(file)
            print(f"✅ Merged & deleted: {file}")

        except Exception as e:
            print(f"❌ Error processing {file}: {e}")

# Move merged file back to the correct location
os.rename(temp_merged_csv, final_merged_csv)

print(f"\n📌 Merged CSV saved at: {final_merged_csv}")


