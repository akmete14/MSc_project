from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBRegressor

df = pd.read_csv('/cluster/project/math/akmete/MSc/preprocessing/df_balanced_groups.csv')
df = df.fillna(0)
print(df.columns)

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

df_balanced = pd.concat(
    resample_sites_iterative(df, site_col='site_id', min_samples=2500, max_samples=3000),
    axis=0
)

# ... (any other resampling code, if used) ...

features = [col for col in df_balanced.columns if col not in ['GPP', 'cluster', 'site_id']]
X = df_balanced[features].values
y = df_balanced['GPP'].values
groups = df_balanced['cluster'].values

logo = LeaveOneGroupOut()
results = []

# Here: specify n_jobs to use multiple CPU cores (e.g., 4).
model = XGBRegressor(
    objective='reg:squarederror',
    random_state=42,
    n_estimators=100,
    max_depth=5,
    learning_rate=0.1,
    n_jobs=4  # <--- Use multiple cores
)

for train_idx, test_idx in logo.split(X, y, groups):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # Scale y (minmax scaling)
    y_train_standard = (y_train - y_train.mean()) / (y_train.max() - y_train.min())
    y_test_standard = (y_test - y_train.mean()) / (y_train.max() - y_train.min())
    
    # Scale X (minmax scaling)
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Fit model on multiple cores
    model.fit(X_train_scaled, y_train_standard)
    
    # Predict and evaluate
    y_pred = model.predict(X_test_scaled)
    mse = mean_squared_error(y_test_standard, y_pred)
    
    results.append({'group_left_out': groups[test_idx][0], 'mse': mse})

results_df = pd.DataFrame(results)
print(results_df)
results_df.to_csv('LOGOCVXGB.csv', index=False)
