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
df = df.dropna(axis=1, how='all')  # Drop columns with all NaN values
df = df.fillna(0) # get rid off NaN values if there are
# Drop unnecessary columns
df = df.drop(columns=['Unnamed: 0'])

# Convert float64 to float32 to save memory
for col in tqdm(df.select_dtypes(include=['float64']).columns, desc="Casting columns"):
    df[col] = df[col].astype('float32')

# Define features and target
feature_columns = [col for col in df.columns if col not in ['GPP', 'site_id', 'cluster']]
target_column = "GPP"

# If we want to parallelize, then SLURM setup, otherwise sequential execution
# --- SLURM Parallelization Setup ---
# Use SLURM_ARRAY_TASK_ID to process one cluster per job.
if 'SLURM_ARRAY_TASK_ID' in os.environ:
    unique_clusters = sorted(df['cluster'].unique())
    index = int(os.environ['SLURM_ARRAY_TASK_ID'])
    test_cluster = unique_clusters[index]
    clusters_to_process = [test_cluster]
else:
    # Otherwise, process all clusters sequentially.
    clusters_to_process = sorted(df['cluster'].unique())

results = {}

# Define for loop for LOGO CV
for cluster in tqdm(clusters_to_process, desc="Processing LOGO clusters"):
    # Define train and test cluster for this fold
    df_train = df[df['cluster'] != cluster].copy()
    df_test  = df[df['cluster'] == cluster].copy()
    
    # Scale feature matrix X
    scaler_X = MinMaxScaler()
    X_train = df_train[feature_columns]
    X_test  = df_test[feature_columns]
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_test_scaled  = scaler_X.transform(X_test)
    
    # get target
    y_train = df_train[target_column]
    y_test  = df_test[target_column]
    
    # scale target
    y_train_min = y_train.min()
    y_train_max = y_train.max()
    if y_train_max - y_train_min == 0:
        y_train_scaled = y_train.values
        y_test_scaled  = y_test.values
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
                         random_state=42,
                         n_jobs=-1)
    model.fit(X_train_scaled, y_train_scaled)
    
    # make prediction and calculate metrics
    y_pred = model.predict(X_test_scaled)
    mse = mean_squared_error(y_test_scaled, y_pred)
    r2  = r2_score(y_test_scaled, y_pred)
    rmse = np.sqrt(mse)
    relative_error = np.mean(np.abs(y_test_scaled - y_pred) / np.abs(y_test_scaled))
    mae = np.mean(np.abs(y_test_scaled - y_pred))
    
    print(f"LOGO - Held-out Cluster {cluster}: MSE={mse:.6f}, R2={r2:.6f}, RMSE={rmse:.6f}, RelError={relative_error:.6f}, MAE={mae:.6f}")
    
    # save results into dataframe
    results[cluster] = {
        'model': model,
        'mse': mse,
        'rmse': rmse,
        'r2_score': r2,
        'relative_error': relative_error,
        'mae': mae
    }

# save results into csv
if 'SLURM_ARRAY_TASK_ID' in os.environ:
    output_filename = f"results_LOGO_{clusters_to_process[0]}.csv"
    results_df = pd.DataFrame([{'cluster': cluster,
                                 'mse': info['mse'],
                                 'rmse': info['rmse'],
                                 'r2_score': info['r2_score'],
                                 'relative_error': info['relative_error'],
                                 'mae': info['mae']}
                                for cluster, info in results.items()])
    results_df.to_csv(output_filename, index=False)
    print(f"Results saved to {output_filename}")
else:
    results_df = pd.DataFrame([{'cluster': cluster,
                                 'mse': info['mse'],
                                 'rmse': info['rmse'],
                                 'r2_score': info['r2_score'],
                                 'relative_error': info['relative_error'],
                                 'mae': info['mae']}
                                for cluster, info in results.items()])
    results_df = results_df.sort_values(by='cluster')
    results_df.to_csv("LOGO_cluster_results.csv", index=False)
    print("All LOGO results saved to LOGO_cluster_results.csv")
