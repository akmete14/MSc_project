import pandas as pd
import numpy as np
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
import xarray as xr
from xgboost import XGBRegressor
from tqdm import tqdm

df_train = pd.read_csv('/cluster/project/math/akmete/MSc/preprocessing/train_data.csv')
df_test = pd.read_csv('/cluster/project/math/akmete/MSc/preprocessing/test_data.csv')
print(len(df_train), len(df_test),df_train.head(),df_test.head(),df_train.columns)

# Reduce memory for both
for col in tqdm(df_train.select_dtypes(include=['float64']).columns):
    df_train[col] = df_train[col].astype('float32')

for col in tqdm(df_test.select_dtypes(include=['float64']).columns):
    df_test[col] = df_test[col].astype('float32')

print("reduced to float32")

feature_cols = [
    col for col in df_train.columns
    if col not in ["GPP", "site_id", "cluster","Unnamed: 0"]
]
target_col = "GPP"

X_train = df_train[feature_cols]
y_train = df_train[target_col]

print("now start scaling")
# Scale X_train and y_train
X_scaler = MinMaxScaler()
X_train_scaled = X_scaler.fit_transform(X_train)
y_train_standard = (y_train - y_train.mean()) / (y_train.max() - y_train.min())
print("scaled")

# Force copy and ensure float32, contiguous layout (otherwise XGBoost complains)
X_train_scaled = np.ascontiguousarray(X_train_scaled, dtype=np.float32)

# Initialize and train model
model = XGBRegressor(objective='reg:squarederror', random_state=42, n_estimators=100, max_depth=5, learning_rate=0.1, n_jobs=-1)
model.fit(X_train_scaled, y_train_standard)
print("model fitted")

# Evaluate
results = []

# Group df_test by 'cluster'
for cluster_id, cluster_data in tqdm(df_test.groupby('cluster')):

    # Extract X and y for the cluster
    X_cluster = cluster_data[feature_cols]
    y_cluster = cluster_data[target_col]
    
    # Scale this cluster's X using the *training* scaler
    X_cluster_scaled = X_scaler.transform(X_cluster)
    y_cluster_standard = (y_cluster - y_train.mean()) / (y_train.max() - y_train.min())
    
    X_cluster_scaled = np.ascontiguousarray(X_cluster_scaled, dtype=np.float32)

    # Predict in the scaled domain
    y_pred = model.predict(X_cluster_scaled)
    
    # Compute MSE (and optionally R^2)
    mse = mean_squared_error(y_cluster_standard, y_pred)
    r2 = r2_score(y_cluster_standard, y_pred)
    
    results.append({
        'cluster': cluster_id,
        'mse': mse,
        'r2': r2,
        'num_samples': len(cluster_data)  # optional info
    })

results_df = pd.DataFrame(results)
results_df.to_csv('Metrics_MinimalDomainShift_XGB.csv')
print("Evaluation Metrics by Cluster:")
print(results_df)