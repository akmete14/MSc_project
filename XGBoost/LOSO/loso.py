# Import libraries
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
df = df.fillna(0) # Fill NaNs with 0 if there are any
# Remove unnecessary columns
df = df.drop(columns=['Unnamed: 0', 'cluster'])

# Convert float64 to float32 to save memory
for col in tqdm(df.select_dtypes(include=['float64']).columns, desc="Casting columns"):
    df[col] = df[col].astype('float32')

# Define features and target
feature_columns = [col for col in df.columns if col not in ['GPP', 'site_id']]
target_column = "GPP"

# Only relevant when parallelizing the job
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

# Initialize results list
results = {}

# Define for loop for running LOSO CV
for test_site in tqdm(sites_to_process, desc="Processing LOSO"):
    # For each fold, define train and test set
    df_train = df[df['site_id'] != test_site].copy()
    df_test  = df[df['site_id'] == test_site].copy()
    
    # Scale features using MinMax scaling. Then apply the scaler from train to the test feature set
    # Fit feature scaler on training data only
    scaler_X = MinMaxScaler()
    X_train = df_train[feature_columns]
    X_test  = df_test[feature_columns]
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_test_scaled  = scaler_X.transform(X_test)
    
    # Extract train and test target
    y_train = df_train[target_column]
    y_test  = df_test[target_column]
    
    # Scale target using statistics from train
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
    
    # Define and train model
    model = XGBRegressor(objective='reg:squarederror',
                         n_estimators=100,
                         max_depth=5,
                         learning_rate=0.1,
                         random_state=42)
    
    model.fit(X_train_scaled, y_train_scaled)
    
    # Get predictions of the model
    y_pred = model.predict(X_test_scaled)
    
    # Calculate some metrics
    mse = mean_squared_error(y_test_scaled, y_pred)
    r2  = r2_score(y_test_scaled, y_pred)
    rmse = np.sqrt(mse)
    relative_error = np.mean(np.abs(y_test_scaled - y_pred) / np.abs(y_test_scaled))
    mae = np.mean(np.abs(y_test_scaled - y_pred))
    
    print(f"LOSO - Test Site {test_site}: MSE={mse:.6f}, R2={r2:.6f}, RMSE={rmse:.6f}, RelError={relative_error:.6f}, MAE={mae:.6f}")
    
    # Save the metrics into results dataframe
    results[test_site] = {
        'model': model,
        'mse': mse,
        'rmse': rmse,
        'r2_score': r2,
        'relative_error': relative_error,
        'mae': mae
    }

# Save results as csv's
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
