# Import required libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.compose import TransformedTargetRegressor
from sklearn.preprocessing import MinMaxScaler
import os
import xarray as xr
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans
from joblib import Parallel, delayed
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

# Define path to data and list of files
path = '/cluster/home/akmete/MSc/Data/'
files = [f for f in os.listdir(path) if f.endswith('.nc')]
files.sort()
assert len(files) % 3 == 0
files = {files[0 + 3 * i][:6]: (files[0 + 3 * i], files[1 + 3 * i], files[2 + 3 * i]) for i in range(len(files) // 3)}

# Function to extract good quality data (that is keep qc=0,1)
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

    # Handle x and y duplication
    if 'x_x' in df_combined.columns and 'y_x' in df_combined.columns:
        df_combined['x'] = df_combined['x_x']  # Choose longitude_x
        df_combined['y'] = df_combined['y_x']  # Choose latitude_x
        df_combined.drop(columns=['x_x', 'y_x', 'x_y', 'y_y'], inplace=True)

    # Extract and process 'DateTime' if present
    if 'time' in df_combined.columns:
        df_combined['time'] = pd.to_datetime(df_combined['time'])
        df_combined['year'] = df_combined['time'].dt.year
        df_combined['month'] = df_combined['time'].dt.month
        df_combined['day'] = df_combined['time'].dt.day
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
    df_clean = df_clean.drop(columns=['Qh', 'Qle', 'Qg', 'rnet', 'CO2air', 'RH', 'Qair', 'Precip', 'Psurf', 'Wind', #'NEE', 'reco',
                                       'WWP', 'SNDPPT', 'SLTPPT', 'PHIKCL', 'PHIHOX', 'ORCDRC', 'CRFVOL', 'CLYPPT', 'CECSOL', 
                                       'BLDFIE', 'AWCtS', 'AWCh3', 'AWCh2', 'AWCh1'], errors='ignore')

    # Sample 25% of the data (so that not OOM)
    df_clean = df_clean.sample(frac=sample_fraction, random_state=42).reset_index(drop=True)

    return df_clean

# Define sites but list format
sites = list(files.keys())
sites = sites[:20] # Take first 20 sites
print(sites)

# Randomize grouping shuffle sites list then group
shuffled_sites = np.random.permutation(sites)
groups = [tuple(shuffled_sites[i:i+2]) for i in range(0, len(shuffled_sites), 2)]


# Dictionaries to store train and test splits
train_splits = {}
test_splits = {}

# Leave-one-out logic: Iterate over each group
for leave_out_idx in range(len(groups)):
    # Lists to store current train and test DataFrames
    train_dataframes = []
    test_dataframes = []

    # Process all groups
    for i, group in enumerate(groups):
        if i == leave_out_idx:
            # Process test group (the left-out group)
            for j in range(2):  # Each tuple has 2 elements
                df_test = initialize_dataframe(*files[group[j]], path=path)
                df_test = df_test.dropna().fillna(0)
                test_dataframes.append(df_test)
        else:
            # Process training groups
            for j in range(2):  # Each tuple has 2 elements
                df_train = initialize_dataframe(*files[group[j]], path=path)
                df_train = df_train.dropna().fillna(0)
                train_dataframes.append(df_train)

    # Combine and store the splits
    train_splits[leave_out_idx] = pd.concat(train_dataframes, axis=0).fillna(0)
    test_splits[leave_out_idx] = pd.concat(test_dataframes, axis=0).fillna(0)

    # Combine train and test to align features
    combined_splits = pd.concat([train_splits[leave_out_idx], test_splits[leave_out_idx]], axis=0)

    # Reindex to separate train and test with aligned features
    train_splits[leave_out_idx] = combined_splits.loc[train_splits[leave_out_idx].index].fillna(0)
    test_splits[leave_out_idx] = combined_splits.loc[test_splits[leave_out_idx].index].fillna(0)


# Results storage
results = []

for i in train_splits:

    # Define train and test
    X_train, y_train = train_splits[i].drop(columns=['GPP']), train_splits[i]['GPP']
    X_test, y_test = test_splits[i].drop(columns=['GPP']), test_splits[i]['GPP']

    # MinMax Scaling
    y_train_standard = (y_train - y_train.mean()) / (y_train.max() - y_train.min())
    y_test_standard = (y_test - y_train.mean()) / (y_train.max() - y_train.min())

    # Define pipeline
    pipeline = Pipeline([
        ('scaler', MinMaxScaler()),  # Scale features
        ('regressor', TransformedTargetRegressor(
            regressor=LinearRegression(),  # Linear regression model (can replace with XGB or LGBM)
            transformer=MinMaxScaler()  # Scale the target variable
        ))
    ])
    pipeline.fit(X_train, y_train_standard)

    # Get the coefficients
    coefficients = pipeline.named_steps['regressor'].regressor_.coef_
    print("Coefficients:", coefficients)

    # Predict GPP
    y_pred = pipeline.predict(X_test)

    # Calculate metrics
    mse = mean_squared_error(y_test_standard, y_pred)
    r2 = r2_score(y_test_standard, y_pred)
    relative_error = np.mean(np.abs(y_test_standard - y_pred) / np.abs(y_test_standard))
    mae = np.mean(np.abs(y_test_standard - y_pred))
    mse_relative_to_avg = mse / np.mean(np.abs(y_test_standard))

    # Print metrics
    #print(f"Mean Squared Error: {mse}")
    #print(f"R2 Score: {r2}")
    #print(f"Relative Error: {relative_error}")
    #print(f"Mean Absolute Error: {mae}")
    #print(f"MSE Relative to Average GPP: {mse_relative_to_avg}")

    # Save results in a dictionary
    results.append({
        'Split': i,
        'MSE': mse,
        'R2': r2,
        'Relative Error': relative_error,
        'Mean Absolute Error': mae,
        'MSE Relative to Avg GPP': mse_relative_to_avg,
        'Coefficients': coefficients
    })

# Convert results to a DataFrame and save as csv
results_df = pd.DataFrame(results)
results_df.to_csv('logowith10.csv', index=False)

print("Results saved to logowith10.csv")