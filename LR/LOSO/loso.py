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
df = df.dropna(axis=1, how='all')  # Drop columns where all values are NaN
df = df.fillna(0) # fill any remaining NaNs with 0
df = df.drop(columns=['Unnamed: 0', 'cluster']) # drop unnecessary columns

# Convert float64 to float32 to save resources
for col in tqdm(df.select_dtypes(include=['float64']).columns, desc="Casting columns"):
    df[col] = df[col].astype('float32')

# Define features and target variable
feature_columns = [col for col in df.columns if col not in ['GPP', 'site_id']]
target_column = "GPP"

# Get all sitenames and sort them alphabetically
unique_sites = sorted(df['site_id'].unique())

# Check for SLURM array task ID to select one left-out site for this job
if 'SLURM_ARRAY_TASK_ID' in os.environ:
    index = int(os.environ['SLURM_ARRAY_TASK_ID'])
    test_site = unique_sites[index]
    sites_to_process = [test_site]
else:
    # If not running as an array job, process all folds sequentially
    sites_to_process = unique_sites

# Initialize results list to save results later
results = {}

# Define LOSO cross-validation loop
for test_site in sites_to_process:
    # For this fold, define train and test set respectively
    train = df[df['site_id'] != test_site]
    test  = df[df['site_id'] == test_site]
    
    # Extract features and target variable
    X_train = train[feature_columns]
    y_train = train[target_column]
    X_test  = test[feature_columns]
    y_test  = test[target_column]
    
    # Scale train features using MinMax scaling and scale test featuers using this scaler
    scaler_X = MinMaxScaler()
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_test_scaled  = scaler_X.transform(X_test)
    
    # Scale train target variable and apply scaling statistics form train to test target
    y_train_min = y_train.min()
    y_train_max = y_train.max()
    if y_train_max - y_train_min == 0:
        y_train_scaled = y_train.values
        y_test_scaled = y_test.values
    else:
        y_train_scaled = (y_train - y_train_min) / (y_train_max - y_train_min)
        y_test_scaled  = (y_test - y_train_min) / (y_train_max - y_train_min)
    
    # Define and fit model
    model = LinearRegression()
    model.fit(X_train_scaled, y_train_scaled)
    
    # Get predictions and metrics
    y_pred_scaled = model.predict(X_test_scaled)
    mse = mean_squared_error(y_test_scaled, y_pred_scaled)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test_scaled, y_pred_scaled)
    relative_error = np.mean(np.abs(y_test_scaled - y_pred_scaled) / np.abs(y_test_scaled))
    mae = np.mean(np.abs(y_test_scaled - y_pred_scaled))
    
    # Store the metrics in the list
    results[test_site] = {'model': model, 'mse': mse, 'rmse': rmse, 'r2': r2, 'relative_error': relative_error, 'mae': mae}
    print(f"Left out site {test_site}: MSE = {mse:.4f}, RMSE = {rmse:.4f}, R2 = {r2:.4f}")

# Save the results as CSV
if 'SLURM_ARRAY_TASK_ID' in os.environ:
    # Save individual fold result when running as a SLURM array job
    output_filename = f"results_LOSO_{sites_to_process[0]}.csv"
    results_df = pd.DataFrame([
        {'site_left_out': site, 'mse': info['mse'], 'rmse': info['rmse'], 'r2': info['r2'], 'relative_error': info['relative_error'], 'mae': info['mae']}
        for site, info in results.items()
    ])
    results_df.to_csv(output_filename, index=False)
    print(f"Results saved to {output_filename}")
else:
    # Combine results for all folds into one CSV file
    results_df = pd.DataFrame([
        {'site_left_out': site, 'mse': info['mse'], 'rmse': info['rmse'], 'r2': info['r2'], 'relative_error': info['relative_error'], 'mae': info['mae']}
        for site, info in results.items()
    ])
    results_df = results_df.sort_values(by='site_left_out')
    results_df.to_csv("results_LOSO_all.csv", index=False)
    print("All LOSO results saved to results_LOSO_all.csv")