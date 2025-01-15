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
# from joblib import Parallel, delayed
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from tqdm import tqdm

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
    df = ds.to_dataframe()
    df_meteo = dr.to_dataframe()
    df_rs = dt.to_dataframe()

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
    #df_combined = df_combined.reset_index(drop=True)

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
    #df_clean = df_clean.sample(frac=sample_fraction, random_state=42)

    return df_clean



# Define cutoff time
cutoff = '2018-02-08 00:45:00'
cutoff = pd.Timestamp(cutoff)



# Define train and test dataframes
df_train = pd.DataFrame()
df_test = pd.DataFrame()

for site in tqdm(sites):
    df = initialize_dataframe(*files[site], path=path)
    
    # Split into train and test based on the cutoff time
    train_data = df[df.index <= cutoff]
    test_data = df[df.index > cutoff]
    
    # Append to the respective DataFrames
    df_train = pd.concat([df_train, train_data], ignore_index=True)
    df_test = pd.concat([df_test, test_data], ignore_index=True)

print(len(df_train), len(df_test))



# Define training and test
X_train, y_train = df_train.drop(columns=['GPP']), df_train['GPP']
X_test, y_test = df_test.drop(columns=['GPP']), df_test['GPP']
# Scale everything
# Scale y (minmax scaling)
y_train_standard = (y_train - y_train.mean()) / (y_train.max() - y_train.min())
y_test_standard = (y_test - y_train.mean()) / (y_train.max() - y_train.min())
# Scale X (minmax scaling)
scaler = MinMaxScaler()
# Fit the scaler on the training data and transform X_train
X_train_scaled = scaler.fit_transform(X_train)
# Transform X_test using the same scaler
X_test_scaled = scaler.transform(X_test)



# Define XGBoost model
model = XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.1)
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
output_file = "AllSites_cutoff_metrics_and_feature_importances.csv"

# Add an identifier column for merging purposes
metrics['Type'] = "Metric"
feature_importances['Type'] = "Feature Importance"

# Merge both DataFrames
combined_df = pd.concat([metrics, feature_importances], axis=0, ignore_index=True)

# Save to CSV
combined_df.to_csv(output_file, index=False)
print(f"Metrics and feature importances saved to {output_file}")