# Import libraries
import pandas as pd
import numpy as np
import os
import xarray as xr
from tqdm import tqdm

path = '/cluster/project/math/akmete/MSc/Data/' # Define the path to the uploaded data, which is in the main directory in 'Data'
files = [f for f in os.listdir(path) if f.endswith('.nc')] # get all files in this folder
files.sort() # sort them (if they aren't already)
assert len(files) % 3 == 0 # each 3 files belong to one site, so group them
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
    # Open data and only keep variables we want
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

    # Handle longitude and latitude duplication (duplicates when we merge)
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

    # Extract good quality data
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
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        # Replace the string column with its encoded numeric version
        df_combined[string_column] = le.fit_transform(df_combined[string_column])


    # Drop quality control columns (not needed anymore)
    columns_to_drop = df_combined.filter(regex='_qc$').columns
    df_clean = df_combined.drop(columns=columns_to_drop)

    # Drop irrelevant columns
    df_clean = df_clean.drop(columns=['Qh', 'Qle', 'Qg', 'rnet', 'CO2air', 'RH', 'Qair', 'Precip', 'Psurf', 'Wind', 'NEE', 'reco', 
                                       'WWP', 'SNDPPT', 'SLTPPT', 'PHIKCL', 'PHIHOX', 'ORCDRC', 'CRFVOL', 'CLYPPT', 'CECSOL', 
                                       'BLDFIE', 'AWCtS', 'AWCh3', 'AWCh2', 'AWCh1', 'latitude', 'longitude', 'x', 'y', 'x_x', 'y_x', 'x_y', 'y_y'], errors='ignore')

    # Sample 5% of the data (if considering subsampling as an option)
    #df_clean = df_clean.sample(frac=sample_fraction, random_state=42).reset_index(drop=True)

    return df_clean

# Add grouping to the clean dataframe (grouping defined as BalancedGrouping algorithm in thesis)
dg = pd.read_csv('grouping_equal_size(1).csv')
dg = dg.drop(columns=['longitude', 'latitude','Unnamed: 0'])
dg.index = dg['site']

dataframes = []
for site in tqdm(sites):
    df = initialize_dataframe(*files[site], path=path)
    df['site_id'] = site
    df['cluster'] = dg['balanced_cluster'][site]
    dataframes.append(df)
print("total dataframe initialized")
# Combine all sites into one dataframe
full_dataframe = pd.concat(dataframes, ignore_index=True)
print("total dataframe concatenated")
full_dataframe.to_csv('df_balanced_groups_onevegindex.csv')