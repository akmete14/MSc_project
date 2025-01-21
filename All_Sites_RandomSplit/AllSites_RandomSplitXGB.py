### WITH(OUT) STRATIFICATION BECAUSE OTHERWISE OOM ###

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
#import dask.dataframe as dd
#from dask_ml.xgboost import XGBRegressor as DXGBRegressor

"""
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

    # Sample 25% of the data
    df_clean = df_clean.sample(frac=sample_fraction, random_state=42).reset_index(drop=True)

    return df_clean



# Define a function which resamples/undersamples so that we get the same amount of data from every site
def resample_sites(df, site_col='site_id', min_samples=10000, max_samples=15000, random_state=42):
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



# Initialize dataframes for all sites
dataframes = []
#sample_fraction = 0.25 # not needed since anyway defined already in the initialize_dataframe function
for site in tqdm(sites):
    df = initialize_dataframe(*files[site], path=path)
    df['site_id'] = site
    dataframes.append(df)
print("total dataframe initialized")
# Combine all sites into one dataframe
full_dataframe = pd.concat(dataframes, ignore_index=True)
print("total dataframe concatenated")
"""

"""""
# Define a function which resamples/undersamples so that we get the same amount of data from every site
def resample_sites(df, site_col='site_id', min_samples=15000, max_samples=20000, random_state=42):
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
    return df_balanced"""""

df = pd.read_csv('/cluster/project/math/akmete/MSc/preprocessing/df_balanced_groups.csv')
print(len(df))


# Convert numeric values to less memory usage columns
for col in df.select_dtypes(include=['float64']).columns:
    df[col] = df[col].astype('float32')


# Perform a random split on the entire combined dataframe
train_dataframe, test_dataframe = train_test_split(df, test_size=0.2, random_state=42)
print("train test split successfull")
print(train_dataframe.head())
print(test_dataframe.head())


#print("now resample")
# Resample in Train but leave Test untouched
# train_dataframe_balanced = resample_sites(df=train_dataframe, site_col='site_id', min_samples=15000, max_samples=20000, random_state=42)
print("now define X and y")
# Define feature and target variables
X_train = train_dataframe.drop(columns=['GPP', 'site_id', 'cluster']) # test_dataframe_balanced
y_train = train_dataframe['GPP'] # train_dataframe_balanced

X_test = test_dataframe.drop(columns=['GPP', 'site_id','cluster'])
y_test = test_dataframe['GPP']


print("now scale")
# Scale y (minmax scaling)
y_train_standard = (y_train - y_train.mean()) / (y_train.max() - y_train.min())
y_test_standard = (y_test - y_train.mean()) / (y_train.max() - y_train.min())

# Scale X (minmax scaling)
scaler = MinMaxScaler()

# Fit the scaler on the training data and transform X_train
X_train_scaled = scaler.fit_transform(X_train)

# Transform X_test using the same scaler
X_test_scaled = scaler.transform(X_test)

X_train_scaled = X_train_scaled.astype('float32')
X_test_scaled = X_test_scaled.astype('float32')
print("scaled")

print("now set up model")
# Set up model
model = XGBRegressor(n_estimators=100, max_depth=4, learning_rate=0.1, n_jobs=-1)
# Train model
model.fit(X_train_scaled, y_train_standard)


print("being trained, make now predictions")
# Make predictions and calculate metrics
y_pred = model.predict(X_test_scaled)
# Calculate metrics
mse = mean_squared_error(y_test_standard, y_pred)
r2 = r2_score(y_test_standard, y_pred)
relative_error = np.mean(np.abs(y_test_standard - y_pred) / np.abs(y_test_standard))
mae = np.mean(np.abs(y_test_standard - y_pred))
rmse = np.sqrt(mse)
print(mse)
print(r2)
print(rmse)
print(mae)

# Get feature importances
importances = model.feature_importances_
feature_importances = pd.DataFrame({
    'Feature': X_train.columns,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

print("Feature Importances:")
print(feature_importances)

metrics = pd.DataFrame({
    "Metric": ["MSE", "R2", "RMSE", "MAE", "Relative Error"],
    "Value": [mse, r2, rmse, mae, relative_error]
})

# Get feature importances
importances = model.feature_importances_
feature_importances = pd.DataFrame({
    'Feature': X_train.columns,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

# Display feature importances
print("Feature Importances:")
print(feature_importances)

# Combine metrics and feature importances into one CSV
output_file = "XGB_AllSites_Random_Split_metrics_and_feature_importances.csv"

# Add an identifier column for merging purposes
metrics['Type'] = "Metric"
feature_importances['Type'] = "Feature Importance"

# Merge both DataFrames
combined_df = pd.concat([metrics, feature_importances], axis=0, ignore_index=True)

# Save to CSV
combined_df.to_csv(output_file, index=False)
print(f"Metrics and feature importances saved to {output_file}")