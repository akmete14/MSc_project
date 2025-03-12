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
df = df.drop(columns=['Unnamed: 0', 'cluster'])  # Drop unnecessary columns

for col in tqdm(df.select_dtypes(include=['float64']).columns, desc="Casting columns"):
    df[col] = df[col].astype('float32')

# Define features and target
feature_columns = [col for col in df.columns if col not in ['GPP', 'site_id']]
target_column = "GPP"

results = {}
predictions_list = []

# Process only the site 'DE-Hai'
target_site = 'DE-Hai'
sites_to_process = [target_site]

# Process the target site
for site in sites_to_process:
    group = df[df['site_id'] == site].copy()
    
    # (Optional) Sort chronologically if you have a date column:
    # group = group.sort_values('date_column')
    
    # Perform an 80/20 chronological split
    n_train = int(len(group) * 0.8)
    train = group.iloc[:n_train]
    test  = group.iloc[n_train:]
    
    # Extract features and target
    X_train = train[feature_columns]
    y_train = train[target_column]
    X_test  = test[feature_columns]
    y_test  = test[target_column]
    
    # Scale features using MinMaxScaler (fit on training data)
    scaler_X = MinMaxScaler()
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_test_scaled  = scaler_X.transform(X_test)
    
    # Scale target based on training data values
    y_train_min = y_train.min()
    y_train_max = y_train.max()
    if y_train_max - y_train_min == 0:
        y_train_scaled = y_train.values
        y_test_scaled  = y_test.values
    else:
        y_train_scaled = (y_train - y_train_min) / (y_train_max - y_train_min)
        y_test_scaled  = (y_test - y_train_min) / (y_train_max - y_train_min)
    
    # Also compute standardized target (using training statistics) for model training
    y_train_standard = (y_train - y_train.min()) / (y_train.max() - y_train.min())
    y_test_standard  = (y_test - y_train.min()) / (y_train.max() - y_train.min())
    
    # Ensure proper formatting (float32 arrays)
    X_train_scaled = np.asarray(X_train_scaled, dtype=np.float32)
    X_test_scaled = np.asarray(X_test_scaled, dtype=np.float32)
    y_train_standard = np.asarray(y_train_standard, dtype=np.float32)
    
    # Define and train the XGBoost model
    model = XGBRegressor(objective='reg:squarederror',
                         n_estimators=100,
                         max_depth=5,
                         learning_rate=0.1,
                         random_state=0)
    model.fit(X_train_scaled, y_train_standard)
    
    # Make predictions on the test data
    y_pred = model.predict(X_test_scaled)
    
    # Calculate performance metrics
    mse = mean_squared_error(y_test_standard, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test_standard, y_pred)
    relative_error = np.mean(np.abs(y_test_standard - y_pred) / np.abs(y_test_standard))
    mae = np.mean(np.abs(y_test_standard - y_pred))
    
    print(f"Site {site}: MSE={mse:.6f}, R2={r2:.6f}, RelError={relative_error:.6f}, MAE={mae:.6f}, RMSE={rmse:.6f}")
    
    # Store the model and performance metrics for the site
    results[site] = {
        'model': model,
        'mse': mse,
        'rmse': rmse,
        'r2_score': r2,
        'relative_error': relative_error,
        'mae': mae
    }
    
    # Save per-sample predictions for the site, with a method identifier "XGB"
    site_predictions = pd.DataFrame({
        'site': site,
        'method': 'XGB',
        'actual': y_test_standard,    # Standardized actual target values
        'predicted': y_pred
    })
    predictions_list.append(site_predictions)

# Combine predictions into a single DataFrame
predictions_df = pd.concat(predictions_list, ignore_index=True)

# Save the overall performance metrics and predictions to CSV files
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