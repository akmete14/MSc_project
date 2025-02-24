import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tqdm import tqdm

# -----------------------
# Load and Preprocess Data
# -----------------------
df = pd.read_csv('/cluster/project/math/akmete/MSc/preprocessing/df_balanced_groups_onevegindex.csv')
df = df.dropna(axis=1, how='all')  # Drop columns where all values are NaN
df = df.fillna(0)
# Drop columns not needed for modeling (we drop "Unnamed: 0" but keep "cluster" for LOGO splitting)
df = df.drop(columns=['Unnamed: 0'])
for col in tqdm(df.select_dtypes(include=['float64']).columns, desc="Casting columns"):
    df[col] = df[col].astype('float32')

# -----------------------
# Define features and target
# -----------------------
# For modeling, we use all columns except "GPP", "site_id", and "cluster" as features.
feature_columns = [col for col in df.columns if col not in ['GPP', 'site_id', 'cluster']]
target_column = "GPP"

# -----------------------
# Create a generator to yield sequence batches (for training)
# -----------------------
def sequence_generator(X, y, seq_len=10, batch_size=32):
    """
    Yields batches of sequences for X and corresponding targets for y.
    X and y are converted to numpy arrays if not already.
    """
    X = np.asarray(X)
    y = np.asarray(y)
    total = len(X) - seq_len
    while True:
        for i in range(0, total, batch_size):
            X_batch = []
            y_batch = []
            for j in range(i, min(i + batch_size, total)):
                X_batch.append(X[j:j+seq_len])
                y_batch.append(y[j+seq_len])
            yield np.array(X_batch), np.array(y_batch)

# -----------------------
# Create Sequences Function (for testing, full-memory)
# -----------------------
def create_sequences(X, y, seq_len=10):
    """
    Build sequences of shape (num_sequences, seq_len, num_features) for X
    and corresponding targets (num_sequences,) for y.
    """
    X = np.asarray(X)
    y = np.asarray(y)
    X_seq, y_seq = [], []
    for i in range(len(X) - seq_len):
        X_seq.append(X[i:i+seq_len])
        y_seq.append(y[i+seq_len])
    return np.array(X_seq), np.array(y_seq)

# -----------------------
# Define output file for results
# -----------------------
output_file = 'results_LOGO.csv'
if not os.path.exists(output_file):
    with open(output_file, 'w') as f:
        f.write("cluster,test_loss,mse,r2,relative_error,mae,rmse\n")

# -----------------------
# Get unique clusters based on 'cluster'
# -----------------------
clusters = sorted(df['cluster'].unique())

# Use SLURM_ARRAY_TASK_ID (if available) to process a single cluster; otherwise, process all.
if 'SLURM_ARRAY_TASK_ID' in os.environ:
    index = int(os.environ['SLURM_ARRAY_TASK_ID'])
    clusters_to_process = [clusters[index]]
else:
    clusters_to_process = clusters

# -----------------------
# LOGO: Process each held-out cluster
# -----------------------
seq_len = 10
batch_size = 32

for cluster in tqdm(clusters_to_process, desc="Processing LOGO clusters"):
    print(f"Processing held-out cluster: {cluster}")
    # LOGO: Training data is all rows NOT from the held-out cluster; test data is from the held-out cluster.
    df_train = df[df['cluster'] != cluster].copy()
    df_test  = df[df['cluster'] == cluster].copy()
    
    # Separate features and target
    X_train_raw = df_train[feature_columns]
    y_train_raw = df_train[target_column]
    X_test_raw  = df_test[feature_columns]
    y_test_raw  = df_test[target_column]
    
    # Scale target using training statistics (min–max scaling)
    y_train_scaled_vals = ((y_train_raw - y_train_raw.mean()) / (y_train_raw.max() - y_train_raw.min())).to_numpy()
    y_test_scaled_vals  = ((y_test_raw - y_train_raw.mean()) / (y_train_raw.max() - y_train_raw.min())).to_numpy()
    
    # Scale features using MinMaxScaler (fit on training data)
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train_raw)
    X_test_scaled  = scaler.transform(X_test_raw)
    
    # For training, use generator to yield batches
    train_gen = sequence_generator(X_train_scaled, y_train_scaled_vals, seq_len=seq_len, batch_size=batch_size)
    steps_per_epoch = (len(X_train_scaled) - seq_len) // batch_size
    
    # For testing, create sequences in memory (assumes held-out cluster is small enough)
    X_test_seq, y_test_seq = create_sequences(X_test_scaled, y_test_scaled_vals, seq_len)
    
    print(f"Cluster {cluster}: Training steps per epoch: {steps_per_epoch}")
    print(f"Cluster {cluster}: X_test_seq shape: {X_test_seq.shape}, y_test_seq shape: {y_test_seq.shape}")
    
    # -----------------------
    # Build LSTM Model
    # -----------------------
    model = Sequential()
    model.add(LSTM(64, input_shape=(seq_len, X_train_scaled.shape[1])))
    model.add(Dense(1))  # Single-output regression
    model.compile(optimizer='adam', loss='mse')
    model.summary()
    
    # Train the model using the generator
    history = model.fit(
        train_gen,
        steps_per_epoch=steps_per_epoch,
        validation_data=(X_test_seq, y_test_seq),
        epochs=20,
        verbose=0
    )
    
    # Evaluate on the test set
    test_loss = model.evaluate(X_test_seq, y_test_seq, verbose=0)
    
    # Get predictions and compute metrics
    y_pred = model.predict(X_test_seq).flatten()
    mse_val = mean_squared_error(y_test_seq, y_pred)
    r2_val = r2_score(y_test_seq, y_pred)
    relative_error = np.mean(np.abs(y_test_seq - y_pred) / np.abs(y_test_seq))
    mae_val = mean_absolute_error(y_test_seq, y_pred)
    rmse_val = np.sqrt(mse_val)
    
    print(f"Metrics for held-out cluster {cluster}: Loss={test_loss:.6f}, MSE={mse_val:.6f}, R2={r2_val:.6f}, Relative Error={relative_error:.6f}, MAE={mae_val:.6f}, RMSE={rmse_val:.6f}")
    
    # Append results to output CSV file
    with open(output_file, 'a') as f:
        f.write(f"{cluster},{test_loss},{mse_val},{r2_val},{relative_error},{mae_val},{rmse_val}\n")
