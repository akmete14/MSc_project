import numpy as np
import pandas as pd

from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

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

df = pd.read_csv('/cluster/project/math/akmete/MSc/preprocessing/df_balanced_groups.csv')
print(df.columns)

# Example resampling code you have:
def resample_sites_iterative(df, site_col='site_id', min_samples=2500, max_samples=3000, random_state=42):
    for site in df[site_col].unique():
        site_data = df[df[site_col] == site]
        site_len = len(site_data)

        if site_len > max_samples:
            yield site_data.sample(n=max_samples, random_state=random_state)
        elif site_len < min_samples:
            yield site_data.sample(n=min_samples, replace=True, random_state=random_state)
        else:
            yield site_data

df_balanced = pd.concat(
    resample_sites_iterative(df, site_col='site_id', min_samples=2500, max_samples=3000),
    axis=0
)

# We now have df_balanced with columns like 'GPP', 'cluster', 'site_id', plus features.

# Extract features and target
features = [col for col in df_balanced.columns if col not in ['GPP', 'cluster', 'site_id']]
X_full = df_balanced[features].values
y_full = df_balanced['GPP'].values
groups_full = df_balanced['cluster'].values  # or 'site_id' if that’s your grouping


# Initialize LOGO
logo = LeaveOneGroupOut()
results = []

seq_len = 48  # example sequence length
num_epochs = 5  # or more, depending on your compute and data size
batch_size = 32

for train_idx, test_idx in logo.split(X_full, y_full, groups_full):
    # Split into train/test
    X_train_raw, X_test_raw = X_full[train_idx], X_full[test_idx]
    y_train_raw, y_test_raw = y_full[train_idx], y_full[test_idx]
    
    # Optional: If train/test contain multiple sites, ensure sorting by time if needed
    
    # Scale X (MinMax or Standard)
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train_raw)
    X_test_scaled = scaler.transform(X_test_raw)

    # Scale y (MinMax)
    y_train_standard = (y_train_time_raw - y_train_time_raw.mean()) / (y_train_time_raw.max() - y_train_time_raw.min())
    y_test_standard = (y_test_time_raw - y_train_time_raw.mean()) / (y_train_time_raw.max() - y_train_time_raw.min())

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
    
    # Train the model
    model.fit(X_train_seq, y_train_seq,
              epochs=num_epochs,
              batch_size=batch_size,
              verbose=0)  # verbose=1 to see training progress
    
    # Predict on test sequences
    y_pred = model.predict(X_test_seq)
    
    # Evaluate (on the same scale we trained on)
    mse = mean_squared_error(y_test_seq, y_pred)
    
    # Record results
    group_left_out = groups_full[test_idx][0]  # or 'site_id' if that's your grouping
    results.append({'group_left_out': group_left_out, 'mse': mse})

# Convert results to DataFrame
results_df = pd.DataFrame(results)
print(results_df)
results_df.to_csv('LOGOCV_LSTM.csv', index=False)
