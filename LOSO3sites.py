import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
import xarray as xr
import os

# Define path to data and access all files
path = '/cluster/home/akmete/MSc/Data/'
files = [f for f in os.listdir(path) if f.endswith('.nc')]  # Get all .nc files
files.sort()
assert len(files) % 3 == 0  # Each site has 3 files
files = {files[0 + 3 * i][:6]: (files[0 + 3 * i], files[1 + 3 * i], files[2 + 3 * i]) for i in range(len(files) // 3)}

# Define the reduced list of sites
sites_reduced = list(files.keys())[:3]  # Select the first 3 sites

def extract_good_quality(df):
    for col in df.columns:
        if col.endswith('_qc'):
            continue
        qc_col = f"{col}_qc"
        if qc_col in df.columns:
            bad_quality_mask = df[qc_col].isin([2, 3])
            df.loc[bad_quality_mask, col] = np.nan
    return df

def initialize_dataframe(file1, file2, file3):
    # Open and preprocess the data
    ds = xr.open_dataset(path + file1, engine='netcdf4')
    dr = xr.open_dataset(path + file2, engine='netcdf4')
    dt = xr.open_dataset(path + file3, engine='netcdf4')

    df = ds.to_dataframe().droplevel(['x', 'y'])
    df_meteo = dr.to_dataframe().droplevel(['x', 'y']).drop(columns=['latitude', 'longitude'], errors='ignore')
    df_rs = dt.to_dataframe().droplevel(['x', 'y']).truncate(after='2021-12-31 23:45:00')

    df_combined = pd.concat([df, df_meteo, df_rs], axis=1)
    df_combined = extract_good_quality(df_combined)

    # Add lagged features, rolling mean, and time features
    df_combined['lag_1'] = df_combined['GPP'].shift(1)
    df_combined['lag_2'] = df_combined['GPP'].shift(2)
    df_combined['rolling_mean'] = df_combined['GPP'].rolling(window=3).mean()
    df_combined['hour'] = df_combined.index.hour
    df_combined['day'] = df_combined.index.day
    df_combined['month'] = df_combined.index.month
    df_combined['year'] = df_combined.index.year

    df_combined = df_combined.dropna(subset=['GPP']).iloc[2:].reset_index(drop=True)

    # One-hot encode categorical variables
    string_column = 'IGBP_veg_short'
    if string_column in df_combined.columns:
        one_hot_encoded = pd.get_dummies(df_combined[string_column], prefix=string_column)
        df_combined = pd.concat([df_combined.drop(columns=[string_column]), one_hot_encoded], axis=1)

    # Drop quality control columns
    qc_columns = df_combined.filter(regex='_qc$').columns
    return df_combined.drop(columns=qc_columns)

# Stack dataframes for selected sites
dfs = []
for i, site in enumerate(sites_reduced):
    print(f"Processing site {i + 1}/{len(sites_reduced)}: {site}")
    df = initialize_dataframe(*files[site])
    df = df.iloc[0:200]
    dfs.append(df)

df_combined = pd.concat(dfs, ignore_index=True)

# Define features and target
X = df_combined.drop(columns=["GPP", "latitude", "longitude"])
y = df_combined["GPP"]

# Define groups for LOGO (based on latitude and longitude)
df_combined["location"] = df_combined["latitude"].round(4).astype(str) + "_" + df_combined["longitude"].round(4).astype(str)
groups = df_combined["location"]

# Initialize LOGO
logo = LeaveOneGroupOut()
# Initialize results list
# Initialize results list
results = []

# Perform Leave-One-Group-Out Cross-Validation
for train_idx, test_idx in tqdm(logo.split(X, y, groups=groups), desc="Cross-validation Progress"):
    # Identify the test location
    test_location = groups.iloc[test_idx].iloc[0]
    print(f"Processing fold with test location: {test_location}")

    # Split the data
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    # Scale the target variable (y) ensuring no data leakage
    y_scaler = MinMaxScaler()
    y_train_scaled = y_scaler.fit_transform(y_train.values.reshape(-1, 1))  # Fit only on training data
    y_test_scaled = y_scaler.transform(y_test.values.reshape(-1, 1))       # Transform test data

    # Define pipeline for features
    pipeline = Pipeline([
        ('scaler', MinMaxScaler()),  # Scale features
        ('xgb', XGBRegressor(
            objective='reg:squarederror',  # XGBoost regression
            max_depth=4,                  # Avoid overfitting
            n_estimators=100,             # Number of boosting rounds
            eta=0.1,                      # Learning rate
            subsample=0.8,                # Subsample ratio
            colsample_bytree=0.8,         # Column sampling ratio
            reg_alpha=1.0,                # L1 regularization
            reg_lambda=1.0                # L2 regularization
        ))
    ])

    # Train the pipeline on scaled data
    pipeline.fit(X_train, y_train_scaled.ravel())

    # Predict on the test set
    y_pred_scaled = pipeline.predict(X_test)

    # Reverse the scaling of predictions and ground truth for evaluation
    y_pred = y_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
    y_test_original = y_scaler.inverse_transform(y_test_scaled).flatten()

    # Calculate performance metric in the original scale
    mse = mean_squared_error(y_test_original, y_pred)

    # Debug: Inspect predictions
    print("Sample y_test_original:", y_test_original[:5])
    print("Sample y_pred:", y_pred[:5])

    # Store results
    results.append({
        "location": test_location,
        "mse": mse
    })

# Convert results to a DataFrame
results_df = pd.DataFrame(results)

# Save results to CSV
results_df.to_csv("LOGO_results.csv", index=False)
print("Cross-validation completed. Results saved to LOGO_results.csv.")


