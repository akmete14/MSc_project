import pandas as pd
from tqdm import tqdm
import numpy as np

df = pd.read_csv('/cluster/project/math/akmete/MSc/preprocessing/df_balanced_groups.csv')
for col in tqdm(df.select_dtypes(include=['float64']).columns):
    df[col] = df[col].astype('float32')
print(df.columns)

# Define the parameters
feature_cols = [col for col in df.columns if col not in ["GPP", "cluster", "site_id","Unnamed: 0"]]
seq_len = 48  # length of each sequence
target_col = "GPP"  # your target (e.g., 'GPP')
output_csv = "sequence_data.csv"

# Preparing for creating sequenced data
rows_for_csv = []

# We’ll group by site, because each site is a separate time series
for site_id, site_df in tqdm(df.groupby("site_id")):
    # Optional: check that all rows have the same 'group' if you know each site belongs to exactly one group
    group_val = site_df["cluster"].iloc[0]
    
    # Convert feature & target columns to NumPy arrays
    # shape = (time_length_site, num_features) for X
    X_site = site_df[feature_cols].values
    # shape = (time_length_site,) for y
    y_site = site_df[target_col].values
    
    # ----------------------------
    # 3) Create sequences for this site
    # ----------------------------
    # We'll generate sequences: X[t : t+seq_len], y[t+seq_len-1] (or some variant)
    # If you want to predict the next step after the sequence, use y[t+seq_len]
    #   but then watch your indexing carefully. Here we predict the last step in the window.

    for start_idx in range(0, len(X_site) - seq_len):
        # Features over [start_idx : start_idx+seq_len]
        X_seq = X_site[start_idx : start_idx + seq_len]
        # Target at the last step in the sequence
        y_val = y_site[start_idx + seq_len]
        
        # Flatten the sequence for CSV storage
        # shape: seq_len * num_features
        X_seq_flat = X_seq.flatten()
        
        # Construct a dictionary that will become one row in the CSV
        row_dict = {
            "site_id": site_id,
            "cluster": group_val,
            "start_index": start_idx  # so we know where the sequence started
        }
        # Add each element of the flattened sequence
        for i, val in enumerate(X_seq_flat):
            row_dict[f"X_{i}"] = val
        
        # Add the target
        row_dict["y"] = y_val
        
        rows_for_csv.append(row_dict)

# ----------------------------
# 4) Build the new DataFrame
# ----------------------------
sequences_df = pd.DataFrame(rows_for_csv)
print(sequences_df.head())
print(sequences_df.tail())
# Save to CSV
sequences_df.to_csv(output_csv, index=False)
print(f"Saved sequences to: {output_csv}")
