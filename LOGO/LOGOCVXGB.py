
import pandas as pd
import numpy as np
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
import xarray as xr
from xgboost import XGBRegressor
from tqdm import tqdm

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
    r2 = r2_score(y_test_standard, y_pred)
    relative_error = np.mean(np.abs(y_test_standard - y_pred) / np.abs(y_test_standard))
    mae = np.mean(np.abs(y_test_standard - y_pred))
    rmse = np.sqrt(mse)
    
    # Store results
    results.append({'group_left_out': groups[test_idx][0], 'mse': mse, 'R2': r2, 'Relative Error': relative_error, 'MAE': mae, 'RMSE': rmse})

# Convert results to a DataFrame for better visualization
import pandas as pd
results_df = pd.DataFrame(results)

# Display results
print(results_df)

results_df.to_csv('LOGOCVXGB_all_metrics.csv', index=False)