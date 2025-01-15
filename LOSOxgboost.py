#https://www.analyticsvidhya.com/blog/2024/01/xgboost-for-time-series-forecasting/
import numpy as np
import pandas as pd
from xgboost import XGBRegressor, plot_importance
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import xarray as xr
import xgboost as xgb
import os
import netCDF4
from tqdm import tqdm
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import Pipeline
from tqdm import tqdm
from sklearn.utils import resample

#Define path to data
path = '/cluster/home/akmete/MSc/Data/'
files = [f for f in os.listdir(path) if f.endswith('.nc')] # all files
files.sort()


#Get the files out of folder which have intermediate NaN's. These are: DE-Lnf, IT-Ro2, SD-Dem, US-Los, US-Syv, US-WCr, US-Wi3 and ZM-Mon
#filt_list = ['DE-Lnf_flux.nc', 'DE-Lnf_meteo.nc', 'DE-Lnf_rs.nc', 'IT-Ro2_flux.nc', 'IT-Ro2_meteo.nc', 'IT-Ro2_rs.nc', 'SD-Dem_flux.nc', 'SD-Dem_meteo.nc', 'SD-Dem_rs.nc',
#'US-Los_flux.nc', 'US-Los_meteo.nc', 'US-Los_rs.nc', 'US-Syv_flux.nc', 'US-Syv_meteo.nc', 'US-Syv_rs.nc', 'US-WCr_flux.nc', 'US-WCr_meteo.nc', 'US-WCr_rs.nc',
#        'US-Wi3_flux.nc', 'US-Wi3_meteo.nc', 'US-Wi3_rs.nc', 'ZM-Mon_flux.nc', 'ZM-Mon_meteo.nc', 'ZM-Mon_rs.nc']

# Remove files that are in list
#filtered_files = [file for file in files if file not in filt_list]
assert len(files) % 3 == 0
files = {files[0+3*i][:6]: (files[0+3*i],files[1+3*i],files[2+3*i]) for i in range(len(files)//3)}

#Define list of sites
sites = list(files.keys())

def extract_good_quality(df):
    for col in df.columns:
        # Skip quality control columns
        if col.endswith('_qc'):
            continue
        # Construct the corresponding quality control column name
        qc_col = f"{col}_qc"
        if qc_col in df.columns:
            # Create a mask for bad quality data
            bad_quality_mask = df[qc_col].isin([2, 3])
            # Set bad quality data points to NaN
            df.loc[bad_quality_mask, col] = np.nan
    return df

def initialize_dataframe(file1, file2, file3):
    #Open data
    ds = xr.open_dataset(path+file1, engine='netcdf4')
    dr = xr.open_dataset(path+file2, engine='netcdf4')
    dt = xr.open_dataset(path+file3, engine='netcdf4')

    #Convert to dataframe
    df = ds.to_dataframe()
    df = pd.DataFrame(df)

    df_meteo = dr.to_dataframe()
    df_meteo = pd.DataFrame(df_meteo)

    df_rs = dt.to_dataframe()
    df_rs = pd.DataFrame(df_rs)

    #Get rid of 'x' and 'y' in multiindex and get rid of 'latitude' and 'longitude' in meteo file
    df = df.droplevel(['x','y'])
    df_meteo = df_meteo.droplevel(['x','y'])
    df_rs = df_rs.droplevel(['x','y'])
    #df = df.drop('latitude', axis=1)
    #df = df.drop('longitude', axis=1)
    df_meteo = df_meteo.drop('latitude', axis=1)
    df_meteo = df_meteo.drop('longitude', axis=1)

    #Truncate rs so that same amount of rows as flux and meteo
    df_rs = df_rs.truncate(after='2021-12-31 23:45:00')

    #Merge dataframes
    df_combined = pd.concat([df, df_meteo, df_rs],axis=1)

    #Mark bad quality data (for all variables which allow for that)
    df_combined = extract_good_quality(df_combined)

    #Add lagged variables and rolling mean
    #df_combined['lag_1'] = df_combined['GPP'].shift(1)
    #df_combined['lag_2'] = df_combined['GPP'].shift(2)
    #df_combined['rolling_mean'] = df_combined['GPP'].rolling(window=3).mean()

    #Get time features
    df_combined['hour'] = df_combined.index.hour
    df_combined['day'] = df_combined.index.day
    df_combined['month'] = df_combined.index.month
    df_combined['year'] = df_combined.index.year

    #Drop those rows where target variable contains NaN (can only drop when not intermediate NaNs - there are 8 sites with intermediate NaNs)
    df_combined = df_combined.dropna(subset=['GPP'])
    
    #Drop first two rows and remove DateTimeIndex and reset it to integer-based index
    df_combined = df_combined.iloc[2:].reset_index(drop=True)

    #Drop categorical variable 'IGBP_veg_short' so that xgboost works properly
    # Identify the string column
    string_column = 'IGBP_veg_short'
    # One-hot encode the string column
    one_hot_encoded = pd.get_dummies(df_combined[string_column], prefix=string_column)
    # Combine the numeric columns with the one-hot encoded columns
    df_encoded = pd.concat([df_combined.drop(columns=[string_column]), one_hot_encoded], axis=1)
    
    #df_combined = df_combined.drop(columns=['IGBP_veg_short'])

    #Consider dropping quality control variables
    columns_to_drop = df_encoded.filter(regex='_qc$').columns
    df_clean = df_encoded.drop(columns=columns_to_drop)
    
    return df_clean

def split_and_scale(df):
    
    #Define train and test sets and convert them to numpy arrays
    train_size = int(len(df) * 0.8)
    train, test = df.iloc[:train_size], df.iloc[train_size:]

    X_train = train.drop(columns=['GPP'])
    y_train = train['GPP']
    X_test = test.drop(columns=['GPP'])
    y_test = test['GPP']

    #Now scale everything and convert to dataframe then to numpy - scalind according to https://stackabuse.com/feature-scaling-data-with-scikit-learn-for-machine-learning-in-python/
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    y_train = y_train.values.reshape(-1, 1)
    y_test = y_test.values.reshape(-1, 1)

    y_scaler = MinMaxScaler()
    y_train_scaled = y_scaler.fit_transform(y_train)
    y_test_scaled = y_scaler.transform(y_test)

    return X_train_scaled, y_train_scaled, X_test_scaled, y_test_scaled

def get_location(df):
    # Features and target
    X = df.drop(columns=["GPP", "latitude", "longitude"])
    y = df["GPP"]

    # Combine latitude and longitude to create a "location group"
    df["location"] = df["latitude"].astype(str) + "_" + df["longitude"].astype(str)
    groups = df["location"]

#Stack dataframes to one dataframe
dfs = []
for i in range(len(sites)):
    print(f"Processing site {i+1}/{len(sites)}: {sites[i]}")
    df = initialize_dataframe(*files[sites[i]])
    df = df.iloc[:200]
    dfs.append(df)
df_combined = pd.concat(dfs, ignore_index=True)
#df_combined = pd.DataFrame()
#for i in range(len(sites)):
    #df = initialize_dataframe(*filtered_files[sites[i]])
    #df_combined = pd.concat([df_combined, df],ignore_index=True)

#Define Feature X and target y
X = df_combined.drop(columns=["GPP", "latitude", "longitude"])
y = df_combined["GPP"]
print("ok1")

#Define new feature "location" for the Leave-One-Out CV later
df_combined["location"] = (df_combined["latitude"].round(4).astype(str) + "_" + df_combined["longitude"].round(4).astype(str))
groups = df_combined["location"]
print("ok2")
#Initialize LOGO
logo = LeaveOneGroupOut()
results = []
print("ok3")

#Go through each location split the data, apply minmax scaler and train the model
# Iterate through each group (location) in the cross-validation
for train_idx, test_idx in tqdm(logo.split(X, y, groups=groups), desc="Cross-validation Progress"):
    # Track progress
    test_location = groups.iloc[test_idx].iloc[0]  # The left-out location
    print(f"Processing fold with test location: {test_location}")

    # Split the data
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    # Optional: Subsample the training data (adjust `n_samples` as needed)[DELETE IF NOT NEEDED - ADDED THIS NEW]
    #X_train, y_train = resample(
    #    X_train, y_train,
    #    replace=False,                # Do not allow replacement
    #    n_samples=int(len(X_train) * 0.1),  # Adjust the fraction (10% here)
    #    random_state=42               # For reproducibility
    #)

    # Step 4: Define the pipeline with MinMaxScaler and XGBoost
    pipeline = Pipeline([
        ('scaler', MinMaxScaler()),  # MinMax scaling
        ('xgb', XGBRegressor())     # XGBoost model
    ])

    # Train the pipeline on the training data
    pipeline.fit(X_train, y_train)

    # Make predictions
    y_pred = pipeline.predict(X_test)

    # Calculate performance metric
    mse = mean_squared_error(y_test, y_pred)
    # Save results including predictions and actual values
    results.append({
        "location": test_location,
        "mse": mse,
        #"y_test": y_test.tolist(),  # Convert to list
        #"y_pred": y_pred.tolist()   # Convert to list
    })

# Step 5: Results as a DataFrame
results_df = pd.DataFrame(results)

# Save the DataFrame to a CSV file
results_df.to_csv(f"results_200withoutlag.csv", index=False)