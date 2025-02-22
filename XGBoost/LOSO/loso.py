import os
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBRegressor
from tqdm import tqdm

# Load and preprocess data
df = pd.read_csv('/cluster/project/math/akmete/MSc/preprocessing/df_balanced_groups_onevegindex.csv')
df = df.dropna(axis=1, how='all')  # Drop columns where all values are NaN
df = df.fillna(0)
# Remove any unnecessary columns (adjust as needed)
df = df.drop(columns=['Unnamed: 0', 'cluster'])

# Cast float64 columns to float32
for col in tqdm(df.select_dtypes(include=['float64']).columns, desc="Casting columns"):
    df[col] = df[col].astype('float32')

# Define features and target
feature_columns = [col for col in df.columns if col not in ['GPP', 'site_id']]
target_column = "GPP"

# --- SLURM Parallelization Setup ---
# If SLURM_ARRAY_TASK_ID is available, use it to select one site to hold out.
if 'SLURM_ARRAY_TASK_ID' in os.environ:
    unique_sites = sorted(df['site_id'].unique())
    index = int(os.environ['SLURM_ARRAY_TASK_ID'])
    test_site = unique_sites[index]
    sites_to_process = [test_site]
else:
    # Otherwise, process all sites sequentially.
    sites_to_process = sorted(df['site_id'].unique())

results = {}

for test_site in tqdm(sites_to_process, desc="Processing LOSO"):
    # Define LOSO split: training is all sites except the held-out test_site.
    df_train = df[df['site_id'] != test_site].copy()
    df_test  = df[df['site_id'] == test_site].copy()
    
    # --- Scaling ---
    # Fit feature scaler on training data only
    scaler_X = MinMaxScaler()
    X_train = df_train[feature_columns]
    X_test  = df_test[feature_columns]
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_test_scaled  = scaler_X.transform(X_test)
    
    # Scale target based on training data values
    y_train = df_train[target_column]
    y_test  = df_test[target_column]
    
    y_train_min = y_train.min()
    y_train_max = y_train.max()
    if y_train_max - y_train_min == 0:
        y_train_scaled = y_train.values
        y_test_scaled = y_test.values
    else:
        y_train_scaled = (y_train - y_train_min) / (y_train_max - y_train_min)
        y_test_scaled  = (y_test - y_train_min) / (y_train_max - y_train_min)
    
    # Convert arrays to float32
    X_train_scaled = np.asarray(X_train_scaled, dtype=np.float32)
    X_test_scaled  = np.asarray(X_test_scaled, dtype=np.float32)
    y_train_scaled = np.asarray(y_train_scaled, dtype=np.float32)
    
    # --- Model Training ---
    model = XGBRegressor(objective='reg:squarederror',
                         n_estimators=100,
                         max_depth=5,
                         learning_rate=0.1,
                         random_state=42)
    
    model.fit(X_train_scaled, y_train_scaled)
    
    # --- Prediction and Metrics ---
    y_pred = model.predict(X_test_scaled)
    
    mse = mean_squared_error(y_test_scaled, y_pred)
    r2  = r2_score(y_test_scaled, y_pred)
    rmse = np.sqrt(mse)
    relative_error = np.mean(np.abs(y_test_scaled - y_pred) / np.abs(y_test_scaled))
    mae = np.mean(np.abs(y_test_scaled - y_pred))
    
    print(f"LOSO - Test Site {test_site}: MSE={mse:.6f}, R2={r2:.6f}, RMSE={rmse:.6f}, RelError={relative_error:.6f}, MAE={mae:.6f}")
    
    results[test_site] = {
        'model': model,
        'mse': mse,
        'rmse': rmse,
        'r2_score': r2,
        'relative_error': relative_error,
        'mae': mae
    }

# --- Save Results ---
if 'SLURM_ARRAY_TASK_ID' in os.environ:
    output_filename = f"results_LOSO_{test_site}.csv"
    results_df = pd.DataFrame([{'site': site,
                                 'mse': info['mse'],
                                 'rmse': info['rmse'],
                                 'r2_score': info['r2_score'],
                                 'relative_error': info['relative_error'],
                                 'mae': info['mae']}
                                for site, info in results.items()])
    results_df.to_csv(output_filename, index=False)
    print(f"Results saved to {output_filename}")
else:
    results_df = pd.DataFrame([{'site': site,
                                 'mse': info['mse'],
                                 'rmse': info['rmse'],
                                 'r2_score': info['r2_score'],
                                 'relative_error': info['relative_error'],
                                 'mae': info['mae']}
                                for site, info in results.items()])
    results_df = results_df.sort_values(by='site')
    results_df.to_csv("LOSO_site_results.csv", index=False)
    print("All LOSO results saved to LOSO_site_results.csv")
