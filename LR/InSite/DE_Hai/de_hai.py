import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler
import numpy as np

# Load and preprocess data
df = pd.read_csv('/cluster/project/math/akmete/MSc/preprocessing/df_balanced_groups_onevegindex.csv')
df = df.dropna(axis=1, how='all')  # Drop columns where all values are NaN
df = df.fillna(0)
df = df.drop(columns=['Unnamed: 0','cluster'])
for col in tqdm(df.select_dtypes(include=['float64']).columns, desc="Casting columns"):
    df[col] = df[col].astype('float32')

# Define features and target
feature_columns = [col for col in df.columns if col not in ['GPP', 'site_id']]
target_column = "GPP"

results = {}
predictions_list = []

# --- Process only the site 'DE-Hai' ---
target_site = 'DE-Hai'
sites_to_process = [target_site]

# Process the target site
for site in sites_to_process:
    group = df[df['site_id'] == site]
    
    # (Optional) Sort by a date column if needed:
    # group = group.sort_values('date_column')
    
    # Perform an 80/20 chronological split
    n_train = int(len(group) * 0.8)
    train = group.iloc[:n_train]
    test  = group.iloc[n_train:]
    
    # Extract features and target variables
    X_train = train[feature_columns]
    y_train = train[target_column]
    X_test  = test[feature_columns]
    y_test  = test[target_column]
    
    # Scale features using MinMaxScaler
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
    
    # Train a linear regression model on the scaled training data
    model = LinearRegression()
    model.fit(X_train_scaled, y_train_scaled)
    
    # Evaluate the model on the scaled test data
    y_pred_scaled = model.predict(X_test_scaled)
    mse = mean_squared_error(y_test_scaled, y_pred_scaled)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test_scaled, y_pred_scaled)
    relative_error = np.mean(np.abs(y_test_scaled - y_pred_scaled) / np.abs(y_test_scaled))
    mae = np.mean(np.abs(y_test_scaled - y_pred_scaled))
    
    # Store the model and performance metrics for the site
    results[site] = {
        'model': model,
        'mse': mse,
        'rmse': rmse,
        'r2_score': r2,
        'relative_error': relative_error,
        'mae': mae
    }
    print(f"Site {site}: MSE = {mse:.6f}")
    
    # --- Save predictions for this site ---
    # Create a DataFrame for the test samples with actual and predicted values.
    # The "method" column is set to "LR" (adjust if adding other methods later).
    site_predictions = pd.DataFrame({
        'site': site,
        'method': 'LR',
        'actual': y_test_scaled,     # Use scaled values (or inverse-transform as needed)
        'predicted': y_pred_scaled
    })
    predictions_list.append(site_predictions)

# Combine predictions into a single DataFrame
predictions_df = pd.concat(predictions_list, ignore_index=True)

# Save predictions and overall results to CSV files for site 'DE-Hai'
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
