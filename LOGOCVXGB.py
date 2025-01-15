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

# Define path to data
path = '/cluster/home/akmete/MSc/Data/'
files = [f for f in os.listdir(path) if f.endswith('.nc')]
files.sort()
assert len(files) % 3 == 0
files = {files[0 + 3 * i][:6]: (files[0 + 3 * i], files[1 + 3 * i], files[2 + 3 * i]) for i in range(len(files) // 3)}

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
    df_clean = df_clean.drop(columns=['NEE', 'reco', 'Qh', 'Qle', 'Qg', 'rnet', 'CO2air', 'RH', 'Qair', 'Precip', 'Psurf', 'Wind',
                                       'WWP', 'SNDPPT', 'SLTPPT', 'PHIKCL', 'PHIHOX', 'ORCDRC', 'CRFVOL', 'CLYPPT', 'CECSOL', 
                                       'BLDFIE', 'AWCtS', 'AWCh3', 'AWCh2', 'AWCh1'], errors='ignore')

    # Sample 25% of the data
    df_clean = df_clean.sample(frac=sample_fraction, random_state=42).reset_index(drop=True)

    return df_clean


#Extract the coordinates of the sites
def extract_site_coordinates(sites, files, path):
    site_coords = []

    for site in sites:
        df = initialize_dataframe(*files[site], path=path)
        longitude = df['longitude'].dropna().iloc[0]
        latitude = df['latitude'].dropna().iloc[0]
        site_coords.append({'site': site, 'longitude': longitude, 'latitude': latitude})

    return pd.DataFrame(site_coords)


# Split data into training and testing sets
def split(df):
    X = df.drop(columns=['GPP'])
    y = df['GPP']
    return train_test_split(X, y, test_size=0.2, random_state=42)

# Step 1a: Assign Groups Based on Longitude and Latitude
def assign_site_groups(site_coords, n_groups=10):
    # Ensure valid longitude and latitude
    coords = site_coords[['longitude', 'latitude']].dropna()

    # Perform clustering
    kmeans = KMeans(n_clusters=n_groups, random_state=42)
    site_coords['group'] = kmeans.fit_predict(coords)

    return site_coords

#Step 1b: Assign site to groups randomly (either use 1a or 1b depending on how to form groups)
def assign_random_groups(sites, n_groups=10):
    """
    Randomly assign each site to one of n_groups.
    """
    # Ensure 'sites' is a flat list
    if not isinstance(sites, list):
        raise ValueError("Input 'sites' must be a flat list of site identifiers.")

    np.random.seed(42)
    group_assignments = np.random.randint(0, n_groups, size=len(sites))

    # Verify lengths match
    if len(sites) != len(group_assignments):
        raise ValueError(f"Length of 'sites' ({len(sites)}) does not match length of 'group_assignments' ({len(group_assignments)}).")

    site_groups = pd.DataFrame({'site': sites, 'group': group_assignments})
    return site_groups



# Step 2: LOGOCV (stack all data and then minmax or standard scale)
def process_group(group, sites, files, coords, path):
    print(f"Processing group: {group}")
    
    # Identify sites in the current group
    left_out_sites = coords[coords['group'] == group]['site'].tolist()
    train_sites = [site for site in sites if site not in left_out_sites]

    # Prepare training data (don't need to split, already on training sites)
    X_train_all, y_train_all = [], []
    for site in train_sites:
        df_train = initialize_dataframe(*files[site], path=path)
        X_train, _, y_train, _ = split(df_train)

        # Drop columns with all NaN values
        X_train = X_train.dropna(axis=1, how='all')

        # Add imputation and scaling
        imputer = SimpleImputer(strategy='mean')  # Replace missing values with mean
        scaler = MinMaxScaler()

        # Impute and scale features
        X_train = pd.DataFrame(imputer.fit_transform(X_train), columns=imputer.feature_names_in_)
        X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)

        # Impute and scale target
        y_train = pd.Series(imputer.fit_transform(y_train.values.reshape(-1, 1)).flatten(), name='GPP')

        X_train_all.append(X_train)
        y_train_all.append(y_train)

    X_train_all = pd.concat(X_train_all, axis=0)
    y_train_all = pd.concat(y_train_all, axis=0)

    # Prepare testing data
    X_test_all, y_test_all = [], []
    for site in left_out_sites:
        df_test = initialize_dataframe(*files[site], path=path)
        X_test, _, y_test, _ = split(df_test)

        # Drop columns with all NaN values (ensure alignment with training)
        X_test = X_test.dropna(axis=1, how='all')

        # Impute and scale features
        X_test = pd.DataFrame(imputer.transform(X_test), columns=imputer.feature_names_in_)
        X_test = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

        # Impute and scale target
        y_test = pd.Series(imputer.transform(y_test.values.reshape(-1, 1)).flatten(), name='GPP')

        X_test_all.append(X_test)
        y_test_all.append(y_test)

    X_test_all = pd.concat(X_test_all, axis=0)
    y_test_all = pd.concat(y_test_all, axis=0)


    # Align test data with training features (double check if this is causing problem when having different features (see chat))
    X_test_all = X_test_all.reindex(columns=X_train_all.columns, fill_value=0)

    # Define the pipeline with only XGBRegressor
    pipeline = Pipeline([
        ('regressor', XGBRegressor(
            n_estimators=100, 
            learning_rate=0.1, 
            max_depth=4, 
            random_state=42
        ))
    ])

    # Train the pipeline
    pipeline.fit(X_train_all, y_train_all)

    # Make predictions
    #y_pred = pipeline.predict(X_test_all)

    # Train and evaluate pipeline
    """pipeline = Pipeline([
        ('scaler', MinMaxScaler()),  # Scale features
        ('regressor', TransformedTargetRegressor(
            regressor=XGBRegressor(
                n_estimators=100, 
                learning_rate=0.1, 
                max_depth=4, 
                random_state=42
            ),
            transformer=MinMaxScaler()  # Scale the target variable
        ))
    ])
    pipeline.fit(X_train_all, y_train_all)"""

    # Print feature importance of model
    """if isinstance(pipeline['regressor'], XGBRegressor):
        feature_importance = pipeline['regressor'].feature_importances_
        feature_importance_df = pd.DataFrame({
            'Feature': X_train_all.columns,
            'Importance': feature_importance
        }).sort_values(by='Importance', ascending=False)
        print(f"Feature importance for XGBoost (Group {group}):")
        print(feature_importance_df)"""
    
    # Print feature importance of model using get_booster() with 'weight'
    regressor = pipeline.named_steps['regressor']  # Access the XGBRegressor from the pipeline
    booster = regressor.get_booster()  # Get the booster object

    # Compute feature importance by weight
    feature_importance = booster.get_score(importance_type='weight')

    # Convert to a sorted DataFrame for better readability
    feature_importance_df = pd.DataFrame({
        'Feature': feature_importance.keys(),
        'Importance': feature_importance.values()
    }).sort_values(by='Importance', ascending=False)

    print(f"Feature importance for XGBoost (Group {group}) by 'weight':")
    print(feature_importance_df)

    y_pred = pipeline.predict(X_test_all)
    mse_group = mean_squared_error(y_test_all, y_pred)
    print(f"MSE for group {group}: {mse_group:.2e}")

    return {'group': group, 'mse': mse_group, 'n_train_sites': len(train_sites), 'n_test_sites': len(left_out_sites)}



if __name__ == "__main__":
    sites = list(files.keys())
    #sites = sites[0:120]  # Example: Using 30 sites for LOGOCV

    # Step 1: Extract site coordinates (not using this step because using random grouping now)
    #site_coords = extract_site_coordinates(sites, files, path)
    #print("Site coordinates:\n", site_coords)

    # Step 2: Assign groups to sites
    site_groups = assign_random_groups(sites, n_groups=10)
    print("Site groups:\n", site_groups)

    # Step 3: Run Leave-One-Group-Out Cross-Validation in parallel
    results = Parallel(n_jobs=10)(
        delayed(process_group)(group, sites, files, site_groups, path) for group in range(10)
    )

    # Step 4: Save Results to CSV
    results_df = pd.DataFrame(results)
    results_df.to_csv('LOGOCVXGB_RandomGroupswithfeatimportance.csv', index=False)
    print("Results saved to results.csv")