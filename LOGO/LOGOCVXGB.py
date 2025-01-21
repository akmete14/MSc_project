
import pandas as pd
import numpy as np
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
import xarray as xr
from xgboost import XGBRegressor
from tqdm import tqdm

# Define a function which resamples/undersamples so that we get the same amount of data from every site
"""def resample_sites(df, site_col='site_id', min_samples=2500, max_samples=3000, random_state=42):
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
def resample_sites_iterative(df, site_col='site_id', min_samples=2500, max_samples=3000, random_state=42):
    for site in df[site_col].unique():
        site_data = df[df[site_col] == site]
        site_len = len(site_data)

        if site_len > max_samples:
            yield site_data.sample(n=max_samples, random_state=random_state)
        elif site_len < min_samples:
            yield site_data.sample(n=min_samples, replace=True, random_state=random_state)
        else:
            yield site_data

df_balanced = pd.concat(resample_sites_iterative(df, site_col='site_id', min_samples=2500, max_samples=3000), axis=0)

df_stratified = resample_sites(df, site_col='site_id', min_samples=2500, max_samples=3000)"""

# Load data and reduce float64 to float32
df = pd.read_csv('/cluster/project/math/akmete/MSc/preprocessing/df_balanced_groups.csv')
for col in tqdm(df.select_dtypes(include=['float64']).columns):
    df[col] = df[col].astype('float32')
print(df.columns)

# Assume df is your dataframe with columns: 'target', 'grouping', and features
features = [col for col in df.columns if col not in ["GPP", "cluster", "site_id","Unnamed: 0"]]
X = df[features].values  # will already be float32 if the DF columns are float32
y = df["GPP"].values     # consider downcasting if it’s float64
groups = df["cluster"].values

# Initialize LeaveOneGroupOut
logo = LeaveOneGroupOut()

# Placeholder for results
results = []

# Example model
model = XGBRegressor(objective='reg:squarederror', random_state=42, n_estimators=100, max_depth=5, learning_rate=0.1, n_jobs=-1)

# Perform LOGO CV
for train_idx, test_idx in tqdm(logo.split(X, y, groups)):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # Scale
    # Scale y (minmax scaling)
    y_train_standard = (y_train - y_train.mean()) / (y_train.max() - y_train.min())
    y_test_standard = (y_test - y_train.mean()) / (y_train.max() - y_train.min())
    
    # Scale X (minmax scaling)
    scaler = MinMaxScaler()
    
    # Fit the scaler on the training data and transform X_train
    X_train_scaled = scaler.fit_transform(X_train)
    
    # Transform X_test using the same scaler
    X_test_scaled = scaler.transform(X_test)

    # Force copy and ensure float32, contiguous layout
    X_train_scaled = np.ascontiguousarray(X_train_scaled, dtype=np.float32)
    X_test_scaled = np.ascontiguousarray(X_test_scaled, dtype=np.float32)
    
    # Fit model
    model.fit(X_train_scaled, y_train_standard)
    
    # Predict and evaluate
    y_pred = model.predict(X_test_scaled)
    mse = mean_squared_error(y_test_standard, y_pred)
    
    # Store results
    results.append({'group_left_out': groups[test_idx][0], 'mse': mse})

# Convert results to a DataFrame for better visualization
import pandas as pd
results_df = pd.DataFrame(results)

# Display results
print(results_df)

results_df.to_csv('LOGOCVXGB_100hrs.csv', index=False)