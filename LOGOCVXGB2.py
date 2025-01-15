import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GroupKFold, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.compose import TransformedTargetRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans
from joblib import Parallel, delayed
from xgboost import XGBRegressor
import os
import xarray as xr


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
    # Load data
    ds = xr.open_dataset(path + file1, engine='netcdf4')
    dr = xr.open_dataset(path + file2, engine='netcdf4')
    dt = xr.open_dataset(path + file3, engine='netcdf4')

    # Merge dataframes
    df = ds.to_dataframe().reset_index()
    df_meteo = dr.to_dataframe().reset_index()
    df_rs = dt.to_dataframe().reset_index()

    df_combined = pd.merge(df, df_meteo, on=['time'], how='outer')
    df_combined = pd.merge(df_combined, df_rs, on=['time'], how='outer')

    # Extract quality data
    df_combined = extract_good_quality(df_combined)
    df_combined = df_combined.dropna(subset=['GPP']).reset_index(drop=True)

    # Handle categorical variables
    if 'IGBP_veg_short' in df_combined.columns:
        one_hot_encoded = pd.get_dummies(df_combined['IGBP_veg_short'], prefix='IGBP', dtype=int)
        df_combined = pd.concat([df_combined.drop(columns=['IGBP_veg_short']), one_hot_encoded], axis=1)

    # Drop unnecessary columns
    irrelevant_cols = ['NEE', 'reco', 'Qh', 'Qle', 'Qg', 'rnet', 'CO2air', 'RH', 'Qair', 'Precip', 'Psurf', 'Wind']
    df_combined = df_combined.drop(columns=[col for col in irrelevant_cols if col in df_combined.columns])

    return df_combined

# Split data into training and testing sets
def split(df):
    X = df.drop(columns=['GPP'])
    y = df['GPP']
    return train_test_split(X, y, test_size=0.2, random_state=42)

# Group assignment
def assign_random_groups(sites, n_groups=10):
    np.random.seed(42)
    group_assignments = np.random.randint(0, n_groups, size=len(sites))
    return pd.DataFrame({'site': sites, 'group': group_assignments})

# LOGOCV processing for each group
def process_group(group, sites, files, coords, path):
    print(f"Processing group: {group}")
    left_out_sites = coords[coords['group'] == group]['site'].tolist()
    train_sites = [site for site in sites if site not in left_out_sites]

    # Prepare training data
    X_train_all, y_train_all = [], []
    for site in train_sites:
        df_train = initialize_dataframe(*files[site], path=path)
        X_train, _, y_train, _ = split(df_train)
        X_train_all.append(X_train)
        y_train_all.append(y_train)

    X_train_all = pd.concat(X_train_all, axis=0)
    y_train_all = pd.concat(y_train_all, axis=0)

    # Prepare testing data
    X_test_all, y_test_all = [], []
    for site in left_out_sites:
        df_test = initialize_dataframe(*files[site], path=path)
        X_test, _, y_test, _ = split(df_test)
        X_test_all.append(X_test)
        y_test_all.append(y_test)

    X_test_all = pd.concat(X_test_all, axis=0)
    y_test_all = pd.concat(y_test_all, axis=0)

    # Align features
    X_test_all = X_test_all.reindex(columns=X_train_all.columns, fill_value=0)

    # Train and evaluate pipeline
    pipeline = Pipeline([
        ('scaler', MinMaxScaler()),  # Scale features
        ('regressor', TransformedTargetRegressor(
            regressor=XGBRegressor(
                n_estimators=100, 
                learning_rate=0.1, 
                max_depth=4, 
                random_state=42
            ),
            transformer=MinMaxScaler()  # Scale target
        ))
    ])
    pipeline.fit(X_train_all, y_train_all)

    # Test on the original target scale
    y_pred_scaled = pipeline.predict(X_test_all)
    y_pred_original = pipeline['regressor'].inverse_transform(y_pred_scaled)
    y_test_original = pipeline['regressor'].inverse_transform(y_test_all)

    mse_original = mean_squared_error(y_test_original, y_pred_original)
    print(f"MSE for group {group} (Original Scale): {mse_original:.2e}")

    return {'group': group, 'mse_original': mse_original, 'n_train_sites': len(train_sites), 'n_test_sites': len(left_out_sites)}

if __name__ == "__main__":
    sites = list(files.keys())
    site_groups = assign_random_groups(sites, n_groups=10)
    print("Site groups:\n", site_groups)

    # Parallel processing for LOGOCV
    results = Parallel(n_jobs=10)(
        delayed(process_group)(group, sites, files, site_groups, path) for group in range(10)
    )

    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv('LOGOCVXGB2_Results.csv', index=False)
    print("Results saved to LOGOCVXGB_Results.csv")
