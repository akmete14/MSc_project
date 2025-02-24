import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
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

path = '/cluster/project/math/akmete/MSc/Data/' #'/cluster/project/math/akmete/MSc/Data'
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
def initialize_dataframe(file1, file2, file3, path):
    # Open data
    ds = xr.open_dataset(path + file1, engine='netcdf4')
    ds = ds[['GPP','GPP_qc','longitude','latitude']]
    dr = xr.open_dataset(path + file2, engine='netcdf4')
    dr = dr[['Tair','Tair_qc','vpd','vpd_qc','SWdown','SWdown_qc','LWdown','LWdown_qc','SWdown_clearsky','IGBP_veg_short']]
    dt = xr.open_dataset(path + file3, engine='netcdf4')
    dt = dt[['LST_TERRA_Day','LST_TERRA_Night','EVI','NIRv','NDWI_band7','LAI','fPAR']]

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

    # Sample 5% of the data
    #df_clean = df_clean.sample(frac=sample_fraction, random_state=42).reset_index(drop=True)

    return df_clean

#df = pd.read_csv('/cluster/project/math/akmete/MSc/preprocessing/df_balanced_groups.csv')

# Define the output file
output_file = 'xgb_extrapolation_in_time_single_site.csv'

# Write the header to the output file
with open(output_file, 'w') as f:
    #f.write(f'# mse trained on {sites[site_idx]}\n')
    f.write("site,mse,r2,relative_error,mae,rmse\n")  # Updated headers

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
    
    split_point = int(len(df) * 0.8) # Calculate the split point
    X_train, y_train = df.drop(columns=['GPP']).iloc[:split_point], df['GPP'].iloc[:split_point]
    X_test, y_test = df.drop(columns=['GPP']).iloc[split_point:], df['GPP'].iloc[split_point:]

    # Scale y (minmax scaling)
    y_train_standard = (y_train - y_train.mean()) / (y_train.max() - y_train.min())
    y_test_standard = (y_test - y_train.mean()) / (y_train.max() - y_train.min())
    
    # Scale X (minmax scaling)
    scaler = MinMaxScaler()
    
    # Fit the scaler on the training data and transform X_train
    X_train_scaled = scaler.fit_transform(X_train)
    
    # Transform X_test using the same scaler
    X_test_scaled = scaler.transform(X_test)

    # Ensure data is properly formatted
    X_train_scaled = np.asarray(X_train_scaled, dtype=np.float32)
    X_test_scaled = np.asarray(X_test_scaled, dtype=np.float32)
    y_train_standard = np.asarray(y_train_standard, dtype=np.float32)

    # Handle missing/infinite values
    X_train_scaled = np.nan_to_num(X_train_scaled, nan=0.0, posinf=1e10, neginf=-1e10)
    X_test_scaled = np.nan_to_num(X_test_scaled, nan=0.0, posinf=1e10, neginf=-1e10)

    # Ensure proper shapes
    if len(X_train_scaled.shape) == 1:
        X_train_scaled = X_train_scaled.reshape(-1, 1)
    if len(X_test_scaled.shape) == 1:
        X_test_scaled = X_test_scaled.reshape(-1, 1)

    # Validate shapes
    print(f"X_train_scaled shape: {X_train_scaled.shape}")
    print(f"X_test_scaled shape: {X_test_scaled.shape}")
    print(f"y_train_standard shape: {y_train_standard.shape}")


    # Define XGBoost model
    model = XGBRegressor(objective='reg:squarederror', n_estimators=100, max_depth=5, learning_rate=0.1)
    
    # Train model
    model.fit(X_train_scaled, y_train_standard)
    
    # Make predictions and calculate metrics
    y_pred = model.predict(X_test_scaled)
    
    # Calculate metrics
    mse = mean_squared_error(y_test_standard, y_pred)
    r2 = r2_score(y_test_standard, y_pred)
    relative_error = np.mean(np.abs(y_test_standard - y_pred) / np.abs(y_test_standard))
    mae = np.mean(np.abs(y_test_standard - y_pred))
    rmse = np.sqrt(mse)
    print(f"MSE, R2, Relative Error, MAE, RMSE: {mse}, {r2}, {relative_error}, {mae}, {rmse} site: {site}")

    # Append to output file
    with open(output_file, 'a') as f:
        f.write(f"{site},{mse},{r2},{relative_error},{mae},{rmse}\n")