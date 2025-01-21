### WITH STRATIFICATION BECAUSE OTHERWISE OOM ###

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
#import random

path = '/cluster/home/akmete/MSc/Data/' #'/cluster/home/akmete/MSc/Data/'
files = [f for f in os.listdir(path) if f.endswith('.nc')]
files.sort()
assert len(files) % 3 == 0
files = {files[0 + 3 * i][:6]: (files[0 + 3 * i], files[1 + 3 * i], files[2 + 3 * i]) for i in range(len(files) // 3)}
sites = list(files.keys())

# Function to extract good quality data
def extract_good_quality(df):
    for col in df.columns:
        if col.endswith('_qc'):
            continue
        qc_col = f"{col}_qc"
        if qc_col in df.columns:
            bad_quality_mask = df[qc_col].isin([2, 3])
            df.loc[bad_quality_mask, col] = np.nan
    return df

    # Function to initialize and clean the dataframe
def initialize_dataframe(file1, file2, file3, path, sample_fraction=0.25):
    # Open data
    ds = xr.open_dataset(path + file1, engine='netcdf4')
    dr = xr.open_dataset(path + file2, engine='netcdf4')
    dt = xr.open_dataset(path + file3, engine='netcdf4')

    # Convert to DataFrame
    df = ds.to_dataframe().reset_index()
    df_meteo = dr.to_dataframe().reset_index()
    df_rs = dt.to_dataframe().reset_index()

    # Merge dataframes on common columns
    df_combined = pd.merge(df, df_meteo, on=['time'], how='outer')
    df_combined = pd.merge(df_combined, df_rs, on=['time'], how='outer')

    # Handle longitude and latitude duplication
    if 'longitude_x' in df_combined.columns and 'latitude_x' in df_combined.columns:
        df_combined['longitude'] = df_combined['longitude_x']  # Choose longitude_x
        df_combined['latitude'] = df_combined['latitude_x']  # Choose latitude_x
        df_combined.drop(columns=['longitude_x', 'latitude_x', 'longitude_y', 'latitude_y'], inplace=True)

    # Extract and process 'DateTime' if present
    if 'time' in df_combined.columns:
        df_combined['time'] = pd.to_datetime(df_combined['time'])
        #df_combined['year'] = df_combined['time'].dt.year
        #df_combined['month'] = df_combined['time'].dt.month
        #df_combined['day'] = df_combined['time'].dt.day
        df_combined['hour'] = df_combined['time'].dt.hour
        df_combined = df_combined.drop(columns=['time'])

    # Truncate rs so that same amount of rows as flux and meteo
    df_combined = extract_good_quality(df_combined)

    # Add lagged variables, rolling mean and time features
    #df_combined['lag_1'] = df_combined['GPP'].shift(1)
    #df_combined['lag_2'] = df_combined['GPP'].shift(2)
    #df_combined['rolling_mean'] = df_combined['GPP'].rolling(window=3).mean()

    # Drop rows with NaN in the target variable
    df_combined = df_combined.dropna(subset=['GPP'])
    df_combined = df_combined.reset_index(drop=True)

    # Handle categorical variables
    string_column = 'IGBP_veg_short'
    if string_column in df_combined.columns:
        one_hot_encoded = pd.get_dummies(df_combined[string_column], prefix=string_column, dtype=int)
        df_combined = pd.concat([df_combined.drop(columns=[string_column]), one_hot_encoded], axis=1)

    # Drop quality control columns
    columns_to_drop = df_combined.filter(regex='_qc$').columns
    df_clean = df_combined.drop(columns=columns_to_drop)

    # Drop irrelevant columns
    df_clean = df_clean.drop(columns=['Qh', 'Qle', 'Qg', 'rnet', 'CO2air', 'RH', 'Qair', 'Precip', 'Psurf', 'Wind', 'NEE', 'reco', 
                                       'WWP', 'SNDPPT', 'SLTPPT', 'PHIKCL', 'PHIHOX', 'ORCDRC', 'CRFVOL', 'CLYPPT', 'CECSOL', 
                                       'BLDFIE', 'AWCtS', 'AWCh3', 'AWCh2', 'AWCh1', 'latitude', 'longitude', 'x', 'y', 'x_x', 'y_x', 'x_y', 'y_y'], errors='ignore')

    # Sample 25% of the data
    df_clean = df_clean.sample(frac=sample_fraction, random_state=42).reset_index(drop=True)

    return df_clean



# Define a function which resamples/undersamples so that we get the same amount of data from every site
def resample_sites(df, site_col='site_id', min_samples=5000, max_samples=10000, random_state=42):
    df_list = []
    for site in df[site_col].unique():
        site_data = df[df[site_col]==site]
        site_len = len(site_data)

        # Undersample if above max
        if site_len > max_samples:
            site_data = site_data.sample(n=max_samples, random_state=random_state)

        # Oversample if below min
        elif site_len < min_samples:
            site_data = site_data.sample(n=min_samples, replace=True, random_state=random_state)

        df_list.append(site_data)

    df_balanced = pd.concat(df_list, axis=0)
    return df_balanced

# Define sequence creating function with length 48 (so one full day)
def create_sequences(X, y, seq_len=48):
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

# Initialize dataframes for all sites
dataframes = []
sample_fraction = .25
for site in tqdm(sites):
    df = initialize_dataframe(*files[site], path=path)
    df['site_id'] = site
    dataframes.append(df)

# Combine all sites into one dataframe
full_dataframe = pd.concat(dataframes, ignore_index=True)

# Perform a random split on the entire combined dataframe
train_dataframe, test_dataframe = train_test_split(full_dataframe, test_size=0.2, random_state=42)
print(train_dataframe.describe())
print(test_dataframe.describe())

# Resample in Train but leave Test untouched
train_dataframe_balanced = resample_sites(df=train_dataframe, site_col='site_id', min_samples=10000, max_samples=15000, random_state=42)



# Define feature and target variables
X_train_time_raw = train_dataframe_balanced.drop(columns=['GPP', 'site_id'])
y_train_time_raw  = train_dataframe_balanced['GPP']
X_test_time_raw  = test_dataframe.drop(columns=['GPP', 'site_id'])
y_test_time_raw  = test_dataframe['GPP']

# Scale y (min-max scaling)
y_train_standard = (y_train_time_raw - y_train_time_raw.mean()) / (y_train_time_raw.max() - y_train_time_raw.min())
y_test_standard = (y_test_time_raw - y_train_time_raw.mean()) / (y_train_time_raw.max() - y_train_time_raw.min())
    
# Scale X (min-max scaling)
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train_time_raw)
X_test_scaled = scaler.transform(X_test_time_raw)
    
# Create sequences
seq_len = 48
X_train_time_seq, y_train_time_seq = create_sequences(X_train_scaled, y_train_standard, seq_len)
X_test_time_seq, y_test_time_seq = create_sequences(X_test_scaled, y_test_standard, seq_len)

# Define LSTM model
model = Sequential()
model.add(LSTM(64, input_shape=(seq_len, X_train_time_seq.shape[2])))  
model.add(Dense(1))  # Single-output regression
model.compile(optimizer='adam', loss='mse')
    
# Train the model
history = model.fit(
    X_train_time_seq, y_train_time_seq,
    validation_split=0.1,
    epochs=20,
    batch_size=32,
    verbose=0
)

# MSE (returned by model.evaluate is the loss function -> MSE in scaled domain)
test_loss = model.evaluate(X_test_time_seq, y_test_time_seq, verbose=0)

# Evaluate the model
y_pred = model.predict(X_test_time_seq).flatten()
mse = mean_squared_error(y_test_time_seq, y_pred)
r2 = r2_score(y_test_time_seq, y_pred)
relative_error = np.mean(np.abs(y_test_time_seq - y_pred) / np.abs(y_test_time_seq))
mae = np.mean(np.abs(y_test_time_seq - y_pred))
rmse = np.sqrt(mse)

# Collect the results in a dictionary
results = {
    'Test Loss': test_loss,
    'MSE': mse,
    'R2': r2,
    'Relative Error': relative_error,
    'MAE': mae,
    'RMSE': rmse,
}

# Convert the dictionary to a DataFrame
results_df = pd.DataFrame([results])

# Save the results to a CSV file
results_df.to_csv('LSTM_AllSites_randomsplit.csv', index=False)

print("Evaluation results saved to 'evaluation_results.csv'")