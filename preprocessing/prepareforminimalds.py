import pandas as pd


df = pd.read_csv('/cluster/project/math/akmete/MSc/preprocessing/df_balanced_groups.csv')


# Define train and test DataFrames
train_list = []
test_list = []

# Group by site_id and perform the 80/20 split for each site
for site_id, group in df.groupby('site_id'):
    # Calculate the split point (80% for train)
    split_index = int(len(group) * 0.8)
    
    # Split into train and test
    train_data = group.iloc[:split_index]
    test_data = group.iloc[split_index:]
    
    # Append to train and test lists
    train_list.append(train_data)
    test_list.append(test_data)

# Combine all train and test data into respective DataFrames
train_df = pd.concat(train_list).reset_index(drop=True)
test_df = pd.concat(test_list).reset_index(drop=True)

# Save to CSV
train_csv_path = "train_data.csv"
test_csv_path = "test_data.csv"

train_df.to_csv(train_csv_path, index=False)
test_df.to_csv(test_csv_path, index=False)

print(f"Train data saved to {train_csv_path}")
print(f"Test data saved to {test_csv_path}")
