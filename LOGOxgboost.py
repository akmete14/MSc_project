import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import resample
from sklearn.cluster import KMeans
from tqdm import tqdm
import os

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

# Function to initialize the DataFrame
def initialize_dataframe(file1, file2, file3, path):
    import xarray as xr
    ds = xr.open_dataset(path + file1, engine='netcdf4')
    dr = xr.open_dataset(path + file2, engine='netcdf4')
    dt = xr.open_dataset(path + file3, engine='netcdf4')
    
    df = ds.to_dataframe().droplevel(['x', 'y'])
    df_meteo = dr.to_dataframe().droplevel(['x', 'y']).drop(['latitude', 'longitude'], axis=1)
    df_rs = dt.to_dataframe().droplevel(['x', 'y']).truncate(after='2021-12-31 23:45:00')
    
    df_combined = pd.concat([df, df_meteo, df_rs], axis=1)
    df_combined = extract_good_quality(df_combined)
    #df_combined['lag_1'] = df_combined['GPP'].shift(1)
    #df_combined['lag_2'] = df_combined['GPP'].shift(2)
    #df_combined['rolling_mean'] = df_combined['GPP'].rolling(window=3).mean()
    df_combined['hour'] = df_combined.index.hour
    df_combined['day'] = df_combined.index.day
    df_combined['month'] = df_combined.index.month
    df_combined['year'] = df_combined.index.year
    df_combined = df_combined.dropna(subset=['GPP']).iloc[2:].reset_index(drop=True)

    # Handle categorical variable
    string_column = 'IGBP_veg_short'
    if string_column in df_combined.columns:
        one_hot_encoded = pd.get_dummies(df_combined[string_column], prefix=string_column)
        df_combined = pd.concat([df_combined.drop(columns=[string_column]), one_hot_encoded], axis=1)
    columns_to_drop = df_combined.filter(regex='_qc$').columns
    df_clean = df_combined.drop(columns=columns_to_drop)
    
    return df_clean

# Function to create groups based on lat-long bins
def create_groups_bins(df, lat_bins, lon_bins):
    df['lat_group'] = pd.cut(df['latitude'], bins=lat_bins, labels=False)
    df['lon_group'] = pd.cut(df['longitude'], bins=lon_bins, labels=False)
    df['group'] = df['lat_group'].astype(str) + "_" + df['lon_group'].astype(str)
    return df

# Function to create groups using clustering
def create_groups_clusters(df, n_clusters):
    coords = df[['latitude', 'longitude']]
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    df['group'] = kmeans.fit_predict(coords)
    return df

# Main processing
path = '/cluster/home/akmete/MSc/Data/'  # Update this to your data path
files = [f for f in os.listdir(path) if f.endswith('.nc')]
files.sort()
assert len(files) % 3 == 0
files = {files[0+3*i][:6]: (files[0+3*i], files[1+3*i], files[2+3*i]) for i in range(len(files)//3)}
sites = list(files.keys())

# Initialize and stack dataframes
dfs = []
for i in range(len(sites)):
    print(f"Processing site {i+1}/{len(sites)}: {sites[i]}")
    df = initialize_dataframe(*files[sites[i]], path=path)
    df = df.iloc[0:200]
    dfs.append(df)
df_combined = pd.concat(dfs, ignore_index=True)

# Create groups (choose one method)
# Option 1: Grid-based binning
lat_bins = np.linspace(df_combined['latitude'].min(), df_combined['latitude'].max(), 10)
lon_bins = np.linspace(df_combined['longitude'].min(), df_combined['longitude'].max(), 10)
df_combined = create_groups_bins(df_combined, lat_bins, lon_bins)

# Option 2: Clustering (comment out grid-based binning if using clustering)
# df_combined = create_groups_clusters(df_combined, n_clusters=10)

# Define features, target, and groups
X = df_combined.drop(columns=["GPP", "latitude", "longitude", "group"])
y = df_combined["GPP"]
groups = df_combined["group"]

# Initialize LOGO
logo = LeaveOneGroupOut()
# Define the pipeline (outside the loop)
pipeline = Pipeline([
    ('scaler', MinMaxScaler()),  # Scale features
    ('xgb', XGBRegressor())     # XGBoost model
])

results = []

for train_idx, test_idx in tqdm(logo.split(X, y, groups=groups), desc="Cross-validation Progress"):
    # Identify the left-out group
    test_group = groups.iloc[test_idx].iloc[0]
    print(f"Processing fold with left-out group: {test_group}")

    # Split the data
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    # Define and fit the pipeline
    pipeline = Pipeline([
        ('scaler', MinMaxScaler()),  # Scale numeric features
        ('xgb', XGBRegressor())     # XGBoost model
    ])
    pipeline.fit(X_train, y_train)

    # Make predictions
    y_pred = pipeline.predict(X_test)

    # Calculate performance metric
    mse = mean_squared_error(y_test, y_pred)

    # Save results
    results.append({
        "group": test_group,  # Include the left-out group in results
        "mse": mse
    })

# Save results to CSV
results_df = pd.DataFrame(results)
results_df.to_csv("LOGO200withoutlag.csv", index=False)

print("Cross-validation completed. Results saved to LOGO200withoutlag.csv.")


""""
# Cross-validation loop with Leave-One-Group-Out
fold_counter = 1
for train_idx, test_idx in logo.split(X, y, groups=groups):
    print(f"Starting fold {fold_counter}...")

    # Split data into training and test sets
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    print(f"Fold {fold_counter}: Train size = {X_train.shape[0]}, Test size = {X_test.shape[0]}.")
    print(X_train.dtypes)
    print(X_train.head())

    # Fit the pipeline on the training data
    print(f"Fitting the pipeline for fold {fold_counter}...")
    pipeline.fit(X_train, y_train)
    print(f"Pipeline fitting completed for fold {fold_counter}.")

    # Predict on the test data
    print(f"Making predictions for fold {fold_counter}...")
    y_pred = pipeline.predict(X_test)
    print(f"Predictions completed for fold {fold_counter}.")

    # Compute MSE for this fold
    mse = mean_squared_error(y_test, y_pred)
    print(f"Fold {fold_counter} MSE: {mse}")

    # Save fold results
    results.append({
        "group": groups.iloc[test_idx].iloc[0],  # The left-out group
        "mse": mse
        #"y_test": y_test.tolist(),  # Save actual test values for analysis
        #"y_pred": y_pred.tolist()   # Save predictions for analysis
    })

    print(f"Results saved for fold {fold_counter}.\n")
    fold_counter += 1

# Convert results into a DataFrame
results_df = pd.DataFrame(results)
print("Cross-validation complete. Compiling results into a DataFrame.")

# Save results to a CSV file
results_df.to_csv("losocv_results.csv", index=False)
print("Results saved to 'losocv_results.csv'.")"""


"""results = []

# Cross-validation with LOGO
for train_idx, test_idx in tqdm(logo.split(X, y, groups=groups), desc="Cross-validation Progress"):
    test_group = groups.iloc[test_idx].iloc[0]
    print(f"Processing fold with test group: {test_group}")

    # Split data
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    # Subsample training data
    X_train, y_train = resample(
        X_train, y_train,
        replace=False, 
        n_samples=int(len(X_train) * 0.1),  # Adjust fraction as needed
        random_state=42
    )

    # Define pipeline
    pipeline = Pipeline([
        ('scaler', MinMaxScaler()),
        ('xgb', XGBRegressor())
    ])

    # Train and evaluate
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)

    results.append({
        "group": test_group,
        "mse": mse
    })

# Save results
results_df = pd.DataFrame(results)
results_df.to_csv("results_lat_lon_groups.csv", index=False)"""
