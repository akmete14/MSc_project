# Put everything into comment so that i can try another code in the same file and not losing the old code which is not working due to OOM
"""import numpy as np
import pandas as pd

from tqdm import tqdm
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Define sequence creating function which is essential for LSTM modeling
def create_sequences(X, y, seq_len=48):
    """
    #Builds a 3D array of shape (num_sequences, seq_len, num_features)
    #and a 1D array of shape (num_sequences,) for targets, 
    #where seq_len is the number of timesteps in each sequence.
    """
    X_seq, y_seq = [], []
    for i in range(len(X) - seq_len):
        X_seq.append(X[i : i + seq_len])
        y_seq.append(y[i + seq_len])
    return np.array(X_seq), np.array(y_seq)

# Read csv (Load in batches to reduce memory usage)
df = pd.read_csv('/cluster/project/math/akmete/MSc/preprocessing/df_balanced_groups.csv')
df.info(memory_usage='deep')
print("loaded dataframe df")
print(len(df))
print(df.columns)

# Convert to numerical columns to float32 to save memory
for col in tqdm(df.select_dtypes(include=["float64"]).columns):
    df[col] = df[col].astype("float32")

print("converted float64 to float32 for saving memory")

# Extract features and target
features = [col for col in df.columns if col not in ['GPP', 'cluster', 'site_id', 'Unnamed: 0']]
X_full = df[features].values
y_full = df['GPP'].values
groups = df['cluster']  # or 'site_id' if that’s your grouping


unique_groups = groups.unique()  # distinct group labels
results = []

print("initialized logo")

seq_len = 48  # example sequence length
num_epochs = 5  # or more, depending on your compute and data size
batch_size = 32

for group_val in tqdm(unique_groups):

    # Define test mask for the group you're leaving out
    test_mask = (groups == group_val)
    train_mask = ~test_mask

    # Extract train data
    X_train_raw = df.loc[train_mask, features].values
    y_train_raw = df.loc[train_mask, 'GPP'].values

    # Extract test data
    X_test_raw = df.loc[test_mask, features].values
    y_test_raw = df.loc[test_mask, 'GPP'].values
        
    # Scale X (MinMax or Standard)
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train_raw)
    X_test_scaled = scaler.transform(X_test_raw)

    # Scale y (MinMax)
    scaler_y = MinMaxScaler()
    # 2) Fit on the training y
    y_train_raw = y_train_raw.reshape(-1, 1)  # needs to be 2D for sklearn
    y_test_raw = y_test_raw.reshape(-1, 1)

    y_train_scaled = scaler_y.fit_transform(y_train_raw)
    y_test_scaled = scaler_y.transform(y_test_raw)

    # 3) Flatten back if you need 1D
    y_train_scaled = y_train_scaled.squeeze()
    y_test_scaled = y_test_scaled.squeeze()

    #y_train_standard = (y_train_raw - y_train_raw.mean()) / (y_train_raw.max() - y_train_raw.min())
    #y_test_standard = (y_test_raw - y_test_raw.mean()) / (y_train_raw.max() - y_train_raw.min())

    # Create sequences
    X_train_seq, y_train_seq = create_sequences(X_train_scaled, y_train_scaled, seq_len=seq_len)
    X_test_seq, y_test_seq = create_sequences(X_test_scaled, y_test_scaled, seq_len=seq_len)
    
    # Check that we still have data after sequence creation
    if len(X_train_seq) == 0 or len(X_test_seq) == 0:
        # Possibly skip if there's not enough data to form a sequence in a group
        continue
    
    # Reshape for LSTM -> (samples, timesteps, features)
    num_features = X_train_seq.shape[2]
    
    # Build an LSTM model
    model = Sequential()
    model.add(LSTM(32, input_shape=(seq_len, num_features)))
    model.add(Dense(1))  # single-value output, e.g., GPP
    model.compile(loss='mse', optimizer='adam')
    

    train_dataset = tf.data.Dataset.from_tensor_slices((X_train_seq, y_train_seq))
    # Batch & prefetch
    train_dataset = train_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    # Fit on the dataset
    model.fit(train_dataset, epochs=num_epochs, verbose=0,workers=4,  # number of CPU cores you want to use
    use_multiprocessing=True)  # or verbose=1 for logs

    # Predict on test sequences
    y_pred = model.predict(X_test_seq)
    
    # Evaluate (on the same scale we trained on)
    mse = mean_squared_error(y_test_seq, y_pred)
    r2 = r2_score(y_test_seq, y_pred)
    relative_error = np.mean(np.abs(y_test_seq - y_pred) / np.abs(y_test_seq))
    mae = np.mean(np.abs(y_test_seq - y_pred))
    rmse = np.sqrt(mse)

    # Record results
    results.append({
    'group_left_out': group_val,
    'mse': mse,
    'r2' : r2,
    'relative_error': relative_error,
    'mae': mae,
    'rmse': rmse
    })

# Convert results to DataFrame
results_df = pd.DataFrame(results)
results_df.to_csv("LOGOCV_LSTM.csv", index=False)"""""

# Code from ChatGPT
import numpy as np
import pandas as pd

from tqdm import tqdm
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

##########################
# 1) CREATE A DATASET PIPELINE FOR SEQUENCES
##########################
def make_sequence_dataset(X, y, seq_len=48, batch_size=32):
    """
    Creates a tf.data.Dataset of (x_seq, y_seq) pairs where:
    - Each x_seq is of shape [seq_len, num_features].
    - y_seq is a single scalar (the last label in that sequence).
    
    This avoids building a huge 3D array in memory.
    """
    # 1. Start from a dataset of (X, y) slices
    ds = tf.data.Dataset.from_tensor_slices((X, y))
    
    # 2. Window the dataset to form sequences of length seq_len
    #    shift=1 -> each new window starts 1 step after the previous.
    #    drop_remainder=True -> ignore partial windows at the end
    ds = ds.window(seq_len, shift=1, drop_remainder=True)
    
    # 3. Each item is now a sub-dataset of length seq_len. We "flatten" it:
    #    x_window is shape (seq_len,) of feature-vectors, y_window is shape (seq_len,)
    #    after calling .batch(seq_len).
    ds = ds.flat_map(lambda x_window, y_window:
                     tf.data.Dataset.zip((
                         x_window.batch(seq_len),
                         y_window.batch(seq_len)
                     )))
    
    # 4. We want (x_seq, y_seq[-1]) if you’re predicting the final step in the sequence.
    #    Or y_seq[-1] could be y_seq[seq_len - 1] – the same thing – if seq_len is the exact size.
    ds = ds.map(lambda x_seq, y_seq: (x_seq, y_seq[-1]))
    
    # 5. Now batch multiple windows together
    ds = ds.batch(batch_size)
    
    # 6. Prefetch for performance
    ds = ds.prefetch(tf.data.AUTOTUNE)
    
    return ds

##########################
# 2) MAIN SCRIPT
##########################

# Load CSV
df = pd.read_csv('/cluster/project/math/akmete/MSc/preprocessing/df_balanced_groups.csv')
df.info(memory_usage='deep')
print("loaded dataframe df")
print(len(df))
print(df.columns)

# Convert float64 -> float32
for col in tqdm(df.select_dtypes(include=["float64"]).columns):
    df[col] = df[col].astype("float32")

print("converted float64 to float32 for saving memory")

features = [col for col in df.columns if col not in ['GPP', 'cluster', 'site_id', 'Unnamed: 0']]
groups = df['cluster']  # or df['site_id']
unique_groups = groups.unique()
results = []

seq_len = 48
num_epochs = 5
batch_size = 32

for group_val in tqdm(unique_groups):
    
    ##########################
    # 2a) SPLIT TRAIN / TEST
    ##########################
    test_mask = (groups == group_val)
    train_mask = ~test_mask
    
    # Extract train data
    X_train_raw = df.loc[train_mask, features].values
    y_train_raw = df.loc[train_mask, 'GPP'].values
    
    # Extract test data
    X_test_raw = df.loc[test_mask, features].values
    y_test_raw = df.loc[test_mask, 'GPP'].values
    
    ##########################
    # 2b) SCALE
    ##########################
    # Scale X
    scaler_X = MinMaxScaler()
    X_train_scaled = scaler_X.fit_transform(X_train_raw)
    X_test_scaled = scaler_X.transform(X_test_raw)
    
    # Scale y
    scaler_y = MinMaxScaler()
    y_train_raw = y_train_raw.reshape(-1, 1)
    y_test_raw = y_test_raw.reshape(-1, 1)
    y_train_scaled = scaler_y.fit_transform(y_train_raw).squeeze()
    y_test_scaled = scaler_y.transform(y_test_raw).squeeze()
    
    # (optional) skip if too small
    if len(X_train_scaled) < seq_len or len(X_test_scaled) < seq_len:
        continue
    
    ##########################
    # 2c) BUILD TF.DATA PIPELINES
    ##########################
    # Instead of create_sequences -> we use make_sequence_dataset
    train_dataset = make_sequence_dataset(
        X_train_scaled, y_train_scaled,
        seq_len=seq_len, batch_size=batch_size
    )
    
    # For test, we do the same. This yields windows for test. 
    # If your test set is big, you want the same approach. 
    # If it's small, you can still do a direct dataset or your old approach:
    test_dataset = make_sequence_dataset(
        X_test_scaled, y_test_scaled,
        seq_len=seq_len, batch_size=batch_size
    )
    
    # Confirm the number of features by looking at first element
    # We won't unroll X_test_seq, so let's just read from the dataset.
    
    ##########################
    # 3) BUILD MODEL
    ##########################
    # We'll guess the # of features from X_train_scaled
    # But to be safe, we can glean from the dataset 
    num_features = X_train_scaled.shape[1]
    
    model = Sequential()
    model.add(LSTM(32, input_shape=(seq_len, num_features)))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam')
    
    # Train
    model.fit(train_dataset, epochs=num_epochs, verbose=0)
    
    ##########################
    # 4) PREDICT & EVALUATE
    ##########################
    # We can directly do model.predict(test_dataset), which yields predictions 
    # for each window in your test set.
    y_pred = model.predict(test_dataset)
    y_pred = y_pred.flatten()  # shape [num_test_windows]
    
    # To compute MSE or R2, you need the matching true labels from the test_dataset
    # Let's collect them:
    y_true_list = []
    for _, y_batch in test_dataset:
        y_true_list.append(y_batch.numpy())
    y_true = np.concatenate(y_true_list).flatten()
    
    # Evaluate
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    relative_error = np.mean(np.abs(y_true - y_pred) / np.abs(y_true))
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(mse)
    
    # Save results
    results.append({
        'group_left_out': group_val,
        'mse': mse,
        'r2': r2,
        'relative_error': relative_error,
        'mae': mae,
        'rmse': rmse
    })

# Save final table
results_df = pd.DataFrame(results)
results_df.to_csv("LOGOCV_LSTM.csv", index=False)
print("Done!")
