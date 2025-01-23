import numpy as np
import pandas as pd

from tqdm import tqdm
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Define sequence creating function which is essential for LSTM modeling
def create_sequences(X, y, seq_len=48):
    """
    Builds a 3D array of shape (num_sequences, seq_len, num_features)
    and a 1D array of shape (num_sequences,) for targets, 
    where seq_len is the number of timesteps in each sequence.
    """
    X_seq, y_seq = [], []
    for i in range(len(X) - seq_len):
        X_seq.append(X[i : i + seq_len])
        y_seq.append(y[i + seq_len])
    return np.array(X_seq), np.array(y_seq)

# Read csv (Load in batches to reduce memory usage)
df = pd.read_csv('/cluster/project/math/akmete/MSc/preprocessing/df_balanced_groups.csv')
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
groups_full = df['cluster'].values  # or 'site_id' if that’s your grouping

# Initialize LOGO
logo = LeaveOneGroupOut()
results = []

print("initialized logo")

seq_len = 48  # example sequence length
num_epochs = 5  # or more, depending on your compute and data size
batch_size = 32

for train_idx, test_idx in tqdm(logo.split(X_full, y_full, groups_full)):
    # Split into train/test
    X_train_raw, X_test_raw = X_full[train_idx], X_full[test_idx]
    y_train_raw, y_test_raw = y_full[train_idx], y_full[test_idx]
        
    # Scale X (MinMax or Standard)
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train_raw)
    X_test_scaled = scaler.transform(X_test_raw)

    # Scale y (MinMax)
    y_train_standard = (y_train_raw - y_train_raw.mean()) / (y_train_raw.max() - y_train_raw.min())
    y_test_standard = (y_test_raw - y_test_raw.mean()) / (y_train_raw.max() - y_train_raw.min())

    # Create sequences
    X_train_seq, y_train_seq = create_sequences(X_train_scaled, y_train_standard, seq_len=seq_len)
    X_test_seq, y_test_seq = create_sequences(X_test_scaled, y_test_standard, seq_len=seq_len)
    
    # Check that we still have data after sequence creation
    if len(X_train_seq) == 0 or len(X_test_seq) == 0:
        # Possibly skip if there's not enough data to form a sequence in a group
        continue
    
    # Reshape for LSTM -> (samples, timesteps, features)
    num_features = X_train_seq.shape[2]
    
    # Build an LSTM model
    model = Sequential()
    model.add(LSTM(64, input_shape=(seq_len, num_features)))
    model.add(Dense(1))  # single-value output, e.g., GPP
    model.compile(loss='mse', optimizer='adam')
    

    train_dataset = tf.data.Dataset.from_tensor_slices((X_train_seq, y_train_seq))
    # Batch & prefetch
    train_dataset = train_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    # Fit on the dataset
    model.fit(train_dataset, epochs=num_epochs, verbose=0)  # or verbose=1 for logs

    # Predict on test sequences
    y_pred = model.predict(X_test_seq)
    
    # Evaluate (on the same scale we trained on)
    mse = mean_squared_error(y_test_seq, y_pred)
    r2 = r2_score(y_test_seq, y_pred)
    relative_error = np.mean(np.abs(y_test_seq - y_pred) / np.abs(y_test_seq))
    mae = np.mean(np.abs(y_test_seq - y_pred))
    rmse = np.sqrt(mse)

    # Record results
    group_left_out = groups_full[test_idx][0]  # or 'site_id' if that's your grouping
    results.append({'group_left_out': group_left_out, 'mse': mse, 'R2': r2, 'Relative Error': relative_error, 'MAE': mae, 'RMSE': rmse})

# Convert results to DataFrame
results_df = pd.DataFrame(results)
print(results_df)
results_df.to_csv('LOGOCV_LSTM.csv', index=False)
