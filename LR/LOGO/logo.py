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
df = df.fillna(0)
df = df.drop(columns=['Unnamed: 0'])
for col in tqdm(df.select_dtypes(include=['float64']).columns, desc="Casting columns"):
    df[col] = df[col].astype('float32')

# Define features and target; exclude site_id and cluster from the predictors.
feature_columns = [col for col in df.columns if col not in ['GPP', 'site_id', 'cluster']]
target_column = "GPP"

# Get sorted unique clusters (assumed there are 10 groups)
unique_clusters = sorted(df['cluster'].unique())

# If running as a SLURM job array, select one cluster based on SLURM_ARRAY_TASK_ID.
if 'SLURM_ARRAY_TASK_ID' in os.environ:
    index = int(os.environ['SLURM_ARRAY_TASK_ID'])
    test_cluster = unique_clusters[index]
    clusters_to_process = [test_cluster]
else:
    # Otherwise, process all clusters sequentially.
    clusters_to_process = unique_clusters

results = {}

for test_cluster in clusters_to_process:
    # Create training data from all clusters except the held-out one.
    train = df[df['cluster'] != test_cluster]
    test  = df[df['cluster'] == test_cluster]
    
    # (Optional) If data has a time column, sort by that column.
    # train = train.sort_values('date_column')
    # test = test.sort_values('date_column')
    
    # Extract features and target variables.
    X_train = train[feature_columns]
    y_train = train[target_column]
    X_test  = test[feature_columns]
    y_test  = test[target_column]
    
    # Scale features using MinMaxScaler, fitted on training data.
    scaler_X = MinMaxScaler()
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_test_scaled  = scaler_X.transform(X_test)
    
    # Scale target variable using min-max scaling based on training data.
    y_train_min = y_train.min()
    y_train_max = y_train.max()
    if y_train_max - y_train_min == 0:
        y_train_scaled = y_train.values
        y_test_scaled  = y_test.values
    else:
        y_train_scaled = (y_train - y_train_min) / (y_train_max - y_train_min)
        y_test_scaled  = (y_test - y_train_min) / (y_train_max - y_train_min)
    
    # Train a linear regression model on the scaled training data.
    model = LinearRegression()
    model.fit(X_train_scaled, y_train_scaled)
    
    # Predict on the scaled test data.
    y_pred_scaled = model.predict(X_test_scaled)
    
    # Calculate performance metrics.
    mse = mean_squared_error(y_test_scaled, y_pred_scaled)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test_scaled, y_pred_scaled)
    relative_error = np.mean(np.abs(y_test_scaled - y_pred_scaled) / np.abs(y_test_scaled))
    mae = np.mean(np.abs(y_test_scaled - y_pred_scaled))
    
    # Save the model and metrics for the held-out group.
    results[test_cluster] = {'model': model, 'mse': mse, 'rmse': rmse, 'r2': r2, 'relative_error': relative_error, 'mae': mae}
    print(f"Left-out cluster {test_cluster}: MSE = {mse:.4f}, RMSE = {rmse:.4f}, R2 = {r2:.4f}")

# Save the results to CSV.
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
