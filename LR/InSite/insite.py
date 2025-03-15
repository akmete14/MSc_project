# Import libraries
import os
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from sklearn.metrics import r2_score

# Load and preprocess data
df = pd.read_csv('/cluster/project/math/akmete/MSc/preprocessing/df_balanced_groups_onevegindex.csv')
df = df.dropna(axis=1, how='all')  # Drop columns where all values are NaN
df = df.fillna(0) # fill NaN with 0 if there are any
df = df.drop(columns=['Unnamed: 0','cluster']) # Drop unnecessary columns

# Convert float64 to float32 to save memory
for col in tqdm(df.select_dtypes(include=['float64']).columns, desc="Casting columns"):
    df[col] = df[col].astype('float32')

# Define features and target
feature_columns = [col for col in df.columns if col not in ['GPP', 'site_id']]
target_column = "GPP"

# Initialize list for results later
results = {}

# SLURM Parallelization Setup
# If running as a SLURM job array, use the SLURM_ARRAY_TASK_ID environment variable to select one site.
if 'SLURM_ARRAY_TASK_ID' in os.environ:
    unique_sites = sorted(df['site_id'].unique())
    index = int(os.environ['SLURM_ARRAY_TASK_ID'])
    sites_to_process = [unique_sites[index]]
else:
    # If not running as a job array, process all sites sequentially.
    sites_to_process = df['site_id'].unique()

# Process each selected site
for site in sites_to_process:
    group = df[df['site_id'] == site]
    
    # Perform an 80/20 chronological split
    n_train = int(len(group) * 0.8)
    train = group.iloc[:n_train]
    test  = group.iloc[n_train:]
    
    # Extract features and target variable
    X_train = train[feature_columns]
    y_train = train[target_column]
    X_test  = test[feature_columns]
    y_test  = test[target_column]
    
    # Scale features using MinMax scaling and apply fitted scaler to test features
    scaler_X = MinMaxScaler()
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_test_scaled  = scaler_X.transform(X_test)
    
    # Scale target variable based on training data values
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
    
    # Get predictions and calculate some metrics
    y_pred_scaled = model.predict(X_test_scaled)
    mse = mean_squared_error(y_test_scaled, y_pred_scaled)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test_scaled, y_pred_scaled)
    relative_error = np.mean(np.abs(y_test_scaled - y_pred_scaled) / np.abs(y_test_scaled))
    mae = np.mean(np.abs(y_test_scaled - y_pred_scaled))


    # Store the results
    results[site] = {'model': model, 'mse': mse, 'rmse': rmse, 'r2_score': r2, 'relative_error': relative_error, 'mae': mae}    
    print(f"Site {site}: MSE = {mse:.6f}")

# Save the results to a csv.
# If running as a SLURM array (one site per job), save with site identifier in filename.
if 'SLURM_ARRAY_TASK_ID' in os.environ:
    output_filename = f"results_site_{sites_to_process[0]}.csv"
    results_df = pd.DataFrame([{'site': site, 'mse': result['mse'], 'rmse': result['rmse'], 'r2_score': result['r2_score'], 'relative_error': result['relative_error'], 'mae': result['mae']} for site, result in results.items()])
    results_df.to_csv(output_filename, index=False)
    print(f"Results saved to {output_filename}")
else:
    # If processing all sites sequentially, combine all into one csv.
    results_df = pd.DataFrame([{'site': site, 'mse': result['mse'], 'rmse': result['rmse'], 'r2_score': result['r2_score'], 'relative_error': result['relative_error'], 'mae': result['mae']} for site, result in results.items()])
    results_df = results_df.sort_values(by='site')
    results_df.to_csv("site_results.csv", index=False)
    print("All site results saved to site_results.csv")