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
df = df.fillna(0) # fill NaNs with 0 if there are any
df = df.drop(columns=['Unnamed: 0','cluster'])  # Drop unnecessary columns

# Convert float64 to float32 to save resources
for col in tqdm(df.select_dtypes(include=['float64']).columns, desc="Casting columns"):
    df[col] = df[col].astype('float32')

# Define feature and target columns
feature_columns = [col for col in df.columns if col not in ['GPP', 'site_id']]
target_column = "GPP"

# Define sequencing functions as needed for LSTM networks
def create_sequences(X, y, seq_len=10):
    """
    Build sequences of shape (num_sequences, seq_len, num_features) for X
    and corresponding targets (num_sequences,) for y.
    """
    # Ensure X and y are numpy arrays
    if isinstance(X, (pd.DataFrame, pd.Series)):
        X = X.to_numpy()
    if isinstance(y, (pd.DataFrame, pd.Series)):
        y = y.to_numpy()
        
    X_seq, y_seq = [], []
    for i in range(len(X) - seq_len):
        X_seq.append(X[i:i+seq_len])
        y_seq.append(y[i+seq_len])
    return np.array(X_seq), np.array(y_seq)

# predefine output file to save results
output_file = 'results_modified.csv'
if not os.path.exists(output_file):
    with open(output_file, 'w') as f:
        f.write("site,test_loss,mse,r2,relative_error,mae,rmse\n")

# Get all unique sitenames to run LOSO
sites = sorted(df['site_id'].unique())

# Use SLURM_ARRAY_TASK_ID if available to process a single site.
if 'SLURM_ARRAY_TASK_ID' in os.environ:
    index = int(os.environ['SLURM_ARRAY_TASK_ID'])
    sites_to_process = [sites[index]]
else:
    sites_to_process = sites

# process DE-Hai site
for site in tqdm(sites_to_process, desc="Processing sites"):
    # read in the data to a csv of this particular site
    df_site = df[df['site_id'] == site].copy()
    print(f"Processing site {site}: shape {df_site.shape}")
    
    # Drop any remaining rows with missing values
    df_site = df_site.dropna(axis=1, how='all').dropna()
    
    # Chronological 80/20 split
    split_index = int(len(df_site) * 0.8)
    df_train = df_site.iloc[:split_index]
    df_test  = df_site.iloc[split_index:]
    
    # Separate features and target using defined variables
    X_train_time_raw = df_train[feature_columns]
    y_train_time_raw = df_train[target_column]
    X_test_time_raw  = df_test[feature_columns]
    y_test_time_raw  = df_test[target_column]
    
    # Scale target using training statistics (min–max scaling)
    y_train_standard = (y_train_time_raw - y_train_time_raw.min()) / (y_train_time_raw.max() - y_train_time_raw.min())
    y_test_standard  = (y_test_time_raw - y_train_time_raw.min()) / (y_train_time_raw.max() - y_train_time_raw.min())
    
    # Scale features using MinMaxScaler (fit on training data)
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train_time_raw)
    X_test_scaled  = scaler.transform(X_test_time_raw)
    
    # Create sequences (LSTM requires 3D input)
    seq_len = 10
    X_train_seq, y_train_seq = create_sequences(X_train_scaled, y_train_standard, seq_len)
    X_test_seq,  y_test_seq  = create_sequences(X_test_scaled,  y_test_standard, seq_len)
    
    print(f"Site {site}: X_train_seq shape: {X_train_seq.shape}, y_train_seq shape: {y_train_seq.shape}")
    print(f"Site {site}: X_test_seq shape: {X_test_seq.shape}, y_test_seq shape: {y_test_seq.shape}")
    
    # Define the LSTM network
    model = Sequential()
    model.add(LSTM(64, input_shape=(seq_len, X_train_seq.shape[2])))
    model.add(Dense(1))  # single-output regression
    model.compile(optimizer='adam', loss='mse')
    model.summary()
    
    # Train model
    history = model.fit(
        X_train_seq, y_train_seq,
        validation_split=0.1,
        epochs=20,
        batch_size=32,
        verbose=0
    )
    
    # Evaluate model
    test_loss = model.evaluate(X_test_seq, y_test_seq, verbose=0)
    
    # Get predictions and compute metrics
    y_pred = model.predict(X_test_seq).flatten()
    mse = mean_squared_error(y_test_seq, y_pred)
    r2 = r2_score(y_test_seq, y_pred)
    relative_error = np.mean(np.abs(y_test_seq - y_pred) / np.abs(y_test_seq))
    mae = mean_absolute_error(y_test_seq, y_pred)
    rmse = np.sqrt(mse)
    
    print(f"Metrics for site {site}: Loss={test_loss:.6f}, MSE={mse:.6f}, R2={r2:.6f}, Relative Error={relative_error:.6f}, MAE={mae:.6f}, RMSE={rmse:.6f}")
    
    # Append results to the output CSV file
    with open(output_file, 'a') as f:
        f.write(f"{site},{test_loss},{mse},{r2},{relative_error},{mae},{rmse}\n")
