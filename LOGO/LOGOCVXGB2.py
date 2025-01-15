# Same but tried to optimize the for loop

from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd

df = pd.read_csv('/cluster/project/math/akmete/MSc/preprocessing/df_balanced_groups.csv')
print(df.columns)
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
    return df_balanced"""
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

df_stratified = resample_sites(df, site_col='site_id', min_samples=2500, max_samples=3000)

# Assume df is your dataframe with columns: 'target', 'grouping', and features
features = [col for col in df_stratified.columns if col not in ['GPP', 'cluster', 'site_id']]
X = df_stratified[features].values
y = df_stratified['GPP'].values
groups = df_stratified['cluster'].values

# Initialize LeaveOneGroupOut
logo = LeaveOneGroupOut()


# Example model
model = XGBRegressor(objective='reg:squarederror', random_state=42, n_estimators=100, max_depth=5, learning_rate=0.1)

results = []

for train_idx, test_idx in logo.split(X, y, groups):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # Scale features
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Scale target
    y_train_standard = (y_train - y_train.mean()) / (y_train.max() - y_train.min())
    y_test_standard = (y_test - y_train.mean()) / (y_train.max() - y_train.min())

    # Train model
    model.fit(X_train_scaled, y_train_standard)

    # Predict and evaluate
    y_pred = model.predict(X_test_scaled)
    mse = mean_squared_error(y_test_standard, y_pred)

    # Append results
    results.append({'group_left_out': groups[test_idx][0], 'mse': mse})

    # Save intermediate results
    pd.DataFrame(results).to_csv('LOGOCVXGB_partial.csv', index=False)