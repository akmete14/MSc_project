import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.compose import TransformedTargetRegressor
from sklearn.preprocessing import MinMaxScaler
import os
import xarray as xr
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans
# from joblib import Parallel, delayed
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import csv

df = pd.read_csv('/cluster/project/math/akmete/MSc/preprocessing/df_balanced_groups.csv')
df = df.drop(columns=['Unnamed: 0', 'cluster'])

# Create Sequences Function
def create_sequences(X, y, seq_len=10):
    """ Builds a 3D array of shape (num_sequences, seq_len, num_features)
        and a 1D array of shape (num_sequences,) for targets.
    """
    # Ensure X, y are numpy arrays with integer-based indexing
    if isinstance(X, pd.DataFrame) or isinstance(X, pd.Series):
        X = X.to_numpy()
    if isinstance(y, pd.DataFrame) or isinstance(y, pd.Series):
        y = y.to_numpy()
    
    X_seq, y_seq = [], []
    for i in range(len(X) - seq_len):
        X_seq.append(X[i:i+seq_len])
        y_seq.append(y[i+seq_len])
    
    return np.array(X_seq), np.array(y_seq)

# Define the output file
output_file = 'baseline_single_site.csv'

# Write the header to the output file (only once, before the loop starts)
with open(output_file, 'w') as f:
    f.write("site,mse,r2,relative_error,mae,rmse\n")  # Header for the results

for site in tqdm(sites):
    # Initialize dataframe and drop NaN values
    df = initialize_dataframe(*files[site], path=path)
    print(f"Initialized dataframe for site {site}: {df.shape}")  # Debug
    df = df.dropna(axis=1, how='all')  # Drop columns where all values are NaN
    df = df.dropna()
    
    # Skip if dataframe is empty
    if df.empty:
        print(f"Dataframe for site {site} is empty after dropping NaN values. Skipping...")
        continue

    # Train-test split
    X = df.drop(columns=['GPP'])
    y = df['GPP']

    # Create sequences first then do random split
    X_seq, y_seq = create_sequences(X, y, seq_len=10)

    X_train_time_raw, X_test_time_raw, y_train_time_raw, y_test_time_raw = train_test_split(X_seq, y_seq, test_size=.2, random_state=42)

    # Scale y (min-max scaling)
    y_train_standard = (y_train_time_raw - y_train_time_raw.min()) / (y_train_time_raw.max() - y_train_time_raw.min())
    y_test_standard = (y_test_time_raw - y_train_time_raw.min()) / (y_train_time_raw.max() - y_train_time_raw.min())
    
    # Scale X (min-max scaling)
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train_time_raw)
    X_test_scaled = scaler.transform(X_test_time_raw)
    
    seq_len = 10
    # Define LSTM model
    model = Sequential()
    model.add(LSTM(64, input_shape=(seq_len, X_train_scaled.shape[2])))  
    model.add(Dense(1))  # Single-output regression
    model.compile(optimizer='adam', loss='mse')
    
    # Train the model
    history = model.fit(
        X_train_scaled, y_train_standard,
        validation_split=0.1,
        epochs=20,
        batch_size=32,
        verbose=0
    )
    
    # Evaluate the model
    y_pred = model.predict(X_test_scaled).flatten()
    mse = mean_squared_error(y_test_standard, y_pred)
    r2 = r2_score(y_test_standard, y_pred)
    relative_error = np.mean(np.abs(y_test_standard - y_pred) / np.abs(y_test_standard))
    mae = np.mean(np.abs(y_test_standard - y_pred))
    rmse = np.sqrt(mse)
    
    print(f"Metrics for site {site}: MSE={mse}, R2={r2}, Relative Error={relative_error}, MAE={mae}, RMSE={rmse}")

    # Append results to the CSV file
    with open(output_file, 'a') as f:
        f.write(f"{site},{mse},{r2},{relative_error},{mae},{rmse}\n")