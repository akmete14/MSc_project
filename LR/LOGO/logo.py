# Import libraries
import os
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm

# Load and preprocess data
df = pd.read_csv('/cluster/project/math/akmete/MSc/preprocessing/df_balanced_groups_onevegindex.csv')
df = df.dropna(axis=1, how='all')  # Drop columns with all NaN values
df = df.fillna(0) # fill any NaNS with 0 if there are any
df = df.drop(columns=['Unnamed: 0']) # drop unnecessary columns

# Convert float64 to float32 to save resources
for col in tqdm(df.select_dtypes(include=['float64']).columns, desc="Casting columns"):
    df[col] = df[col].astype('float32')

# Define features and target variable
feature_columns = [col for col in df.columns if col not in ['GPP', 'site_id', 'cluster']]
target_column = "GPP"

# Get all unique clusters (sorted 0 to 9)
unique_clusters = sorted(df['cluster'].unique())

# For parallelizing via SLURM
# If running as a SLURM job array, select one cluster based on SLURM_ARRAY_TASK_ID.
if 'SLURM_ARRAY_TASK_ID' in os.environ:
    index = int(os.environ['SLURM_ARRAY_TASK_ID'])
    test_cluster = unique_clusters[index]
    clusters_to_process = [test_cluster]
else:
    # Otherwise, process all clusters sequentially.
    clusters_to_process = unique_clusters

# Initialize results list to save results later on
results = {}

# Define LOGO cross-validation loop
for test_cluster in clusters_to_process:
    # For this fold, define train and test set
    train = df[df['cluster'] != test_cluster]
    test  = df[df['cluster'] == test_cluster]
    
    # Extract features and target variable
    X_train = train[feature_columns]
    y_train = train[target_column]
    X_test  = test[feature_columns]
    y_test  = test[target_column]
    
    # Scale train features using MinMax scaling and apply the scaler to the test features
    scaler_X = MinMaxScaler()
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_test_scaled  = scaler_X.transform(X_test)
    
    # Scale train target variable and use the statistics to scale test target variable
    y_train_min = y_train.min()
    y_train_max = y_train.max()
    if y_train_max - y_train_min == 0:
        y_train_scaled = y_train.values
        y_test_scaled  = y_test.values
    else:
        y_train_scaled = (y_train - y_train_min) / (y_train_max - y_train_min)
        y_test_scaled  = (y_test - y_train_min) / (y_train_max - y_train_min)
    
    # Define and train model
    model = LinearRegression()
    model.fit(X_train_scaled, y_train_scaled)
    
    # Make predictions
    y_pred_scaled = model.predict(X_test_scaled)
    
    # Calculate the metrics
    mse = mean_squared_error(y_test_scaled, y_pred_scaled)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test_scaled, y_pred_scaled)
    relative_error = np.mean(np.abs(y_test_scaled - y_pred_scaled) / np.abs(y_test_scaled))
    mae = np.mean(np.abs(y_test_scaled - y_pred_scaled))
    
    # Save the results of th cross-validation
    results[test_cluster] = {'model': model, 'mse': mse, 'rmse': rmse, 'r2': r2, 'relative_error': relative_error, 'mae': mae}
    print(f"Left-out cluster {test_cluster}: MSE = {mse:.4f}, RMSE = {rmse:.4f}, R2 = {r2:.4f}")

# Save the results to csv (for each fold we get one csv)
if 'SLURM_ARRAY_TASK_ID' in os.environ:
    # When running as a SLURM array job, save the result for the single held-out cluster.
    output_filename = f"results_LOGO_{clusters_to_process[0]}.csv"
    results_df = pd.DataFrame([
        {'cluster_left_out': cluster, 'mse': info['mse'], 'rmse': info['rmse'], 'r2': info['r2'], 'relative_error': info['relative_error'], 'mae': info['mae']}
        for cluster, info in results.items()
    ])
    results_df.to_csv(output_filename, index=False)
    print(f"Results saved to {output_filename}")
else:
    # When processing all clusters sequentially, combine results into one CSV file.
    results_df = pd.DataFrame([
        {'cluster_left_out': cluster, 'mse': info['mse'], 'rmse': info['rmse'], 'r2': info['r2'], 'relative_error': info['relative_error'], 'mae': info['mae']}
        for cluster, info in results.items()
    ])
    results_df = results_df.sort_values(by='cluster_left_out')
    results_df.to_csv("results_LOGO_all.csv", index=False)
    print("All LOGO results saved to results_LOGO_all.csv")