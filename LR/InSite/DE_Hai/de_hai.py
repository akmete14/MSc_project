# This script follows the same logic as insite.py, but here we are only interested in site DE-Hai (for plots)
# Import libraries
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler
import numpy as np

# Load and preprocess data
df = pd.read_csv('/cluster/project/math/akmete/MSc/preprocessing/df_balanced_groups_onevegindex.csv')
df = df.dropna(axis=1, how='all')  # Drop columns where all values are NaN
df = df.fillna(0) # fill NaNs with 0 if there are any
df = df.drop(columns=['Unnamed: 0','cluster']) # drop unnecessary columns

# Convert float64 to float32 to save resources
for col in tqdm(df.select_dtypes(include=['float64']).columns, desc="Casting columns"):
    df[col] = df[col].astype('float32')

# Define features and target
feature_columns = [col for col in df.columns if col not in ['GPP', 'site_id']]
target_column = "GPP"

# Initiliaze result and predictions list
results = {}
predictions_list = []

# Define desired site to be executed
target_site = 'DE-Hai'
sites_to_process = [target_site]

# Process this site
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
    
    # Scale features using MinMax and scale test features with scaler fitted on train
    scaler_X = MinMaxScaler()
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_test_scaled  = scaler_X.transform(X_test)
    
    # Scale target variable based on training target
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
    
    # Store the results
    results[site] = {
        'model': model,
        'mse': mse,
        'rmse': rmse,
        'r2_score': r2,
        'relative_error': relative_error,
        'mae': mae
    }
    print(f"Site {site}: MSE = {mse:.6f}")
    
    # Create dataframe with actual data and predicted values and safe them
    site_predictions = pd.DataFrame({
        'site': site,
        'method': 'LR',
        'actual': y_test_scaled,
        'predicted': y_pred_scaled
    })
    predictions_list.append(site_predictions)

# Combine predictions into one dataframe
predictions_df = pd.concat(predictions_list, ignore_index=True)

# Save predictions and results for DE-Hai into a csv
results_df = pd.DataFrame([{
    'site': site,
    'mse': res['mse'],
    'rmse': res['rmse'],
    'r2_score': res['r2_score'],
    'relative_error': res['relative_error'],
    'mae': res['mae']
} for site, res in results.items()])

results_filename = f"results_site_{target_site}.csv"
predictions_filename = f"predictions_site_{target_site}.csv"
results_df.to_csv(results_filename, index=False)
predictions_df.to_csv(predictions_filename, index=False)
print(f"Results saved to {results_filename}")
print(f"Predictions saved to {predictions_filename}")
