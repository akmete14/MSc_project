import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tqdm import tqdm

# Load and preprocess data
df = pd.read_csv('/cluster/project/math/akmete/MSc/preprocessing/df_balanced_groups_onevegindex.csv')
df = df.dropna(axis=1, how='all')  # Drop columns where all values are NaN
df = df.fillna(0) # fill NaN with zero if there are any
df = df.drop(columns=['Unnamed: 0', 'cluster']) # Drop unnecessary columns

# Convert float64 to float32
for col in tqdm(df.select_dtypes(include=['float64']).columns, desc="Casting columns"):
    df[col] = df[col].astype('float32')

# Define feature and target columns
feature_columns = [col for col in df.columns if col not in ['GPP', 'site_id']]
target_column = "GPP"

# Define generator function
def sequence_generator(X, y, seq_len=10, batch_size=32):
    """
    Yields batches of sequences for X and corresponding targets for y.
    X and y are converted to numpy arrays if they are not already.
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

# Define sequencing function
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

# Define where to save the results
output_file = 'results_LOSO_modified.csv'
if not os.path.exists(output_file):
    with open(output_file, 'w') as f:
        f.write("site,test_loss,mse,r2,relative_error,mae,rmse\n")

# Get all unique sitenames
sites = sorted(df['site_id'].unique())

# Use SLURM_ARRAY_TASK_ID (if available) to process a single site; otherwise, process all.
if 'SLURM_ARRAY_TASK_ID' in os.environ:
    index = int(os.environ['SLURM_ARRAY_TASK_ID'])
    sites_to_process = [sites[index]]
else:
    sites_to_process = sites

# Define LSTM network parameters
seq_len = 10
batch_size = 32

# Define LOSO cross-validation
for site in tqdm(sites_to_process, desc="Processing LOSO sites"):
    print(f"Processing held-out site: {site}")
    # Define train and test for this fold
    df_train = df[df['site_id'] != site].copy()
    df_test  = df[df['site_id'] == site].copy()
    
    # Separate features and target using defined variables.
    X_train_raw = df_train[feature_columns]
    y_train_raw = df_train[target_column]
    X_test_raw  = df_test[feature_columns]
    y_test_raw  = df_test[target_column]
    
    # Scale target using training statistics (min–max scaling)
    # Convert the scaled target to a numpy array to ensure integer-based indexing.
    y_train_scaled_vals = ((y_train_raw - y_train_raw.min()) / (y_train_raw.max() - y_train_raw.min())).to_numpy()
    y_test_scaled_vals  = ((y_test_raw - y_train_raw.min()) / (y_train_raw.max() - y_train_raw.min())).to_numpy()
    
    # Scale features using MinMaxScaler (fit on training data)
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train_raw)
    X_test_scaled  = scaler.transform(X_test_raw)
    
    # For training, use the generator to yield batches
    train_gen = sequence_generator(X_train_scaled, y_train_scaled_vals, seq_len=seq_len, batch_size=batch_size)
    steps_per_epoch = (len(X_train_scaled) - seq_len) // batch_size
    
    # For testing, we assume the held-out site's data is small enough to build sequences in memory.
    X_test_seq, y_test_seq = create_sequences(X_test_scaled, y_test_scaled_vals, seq_len)
    
    print(f"Site {site}: Training steps per epoch: {steps_per_epoch}")
    print(f"Site {site}: X_test_seq shape: {X_test_seq.shape}, y_test_seq shape: {y_test_seq.shape}")
    
    # Define LSTM network
    model = Sequential()
    model.add(LSTM(64, input_shape=(seq_len, X_train_scaled.shape[1])))
    model.add(Dense(1))  # single-output regression
    model.compile(optimizer='adam', loss='mse')
    model.summary()
    
    # Train model
    history = model.fit(
        train_gen,
        steps_per_epoch=steps_per_epoch,
        validation_data=(X_test_seq, y_test_seq),
        epochs=20,
        verbose=0
    )
    
    # Evaluate model
    test_loss = model.evaluate(X_test_seq, y_test_seq, verbose=0)
    
    # Get predictions and compute additional metrics
    y_pred = model.predict(X_test_seq).flatten()
    mse_val = mean_squared_error(y_test_seq, y_pred)
    r2_val = r2_score(y_test_seq, y_pred)
    relative_error = np.mean(np.abs(y_test_seq - y_pred) / np.abs(y_test_seq))
    mae_val = mean_absolute_error(y_test_seq, y_pred)
    rmse_val = np.sqrt(mse_val)
    
    print(f"Metrics for held-out site {site}: Loss={test_loss:.6f}, MSE={mse_val:.6f}, R2={r2_val:.6f}, Relative Error={relative_error:.6f}, MAE={mae_val:.6f}, RMSE={rmse_val:.6f}")
    
    # Append results to the output CSV file
    with open(output_file, 'a') as f:
        f.write(f"{site},{test_loss},{mse_val},{r2_val},{relative_error},{mae_val},{rmse_val}\n")
