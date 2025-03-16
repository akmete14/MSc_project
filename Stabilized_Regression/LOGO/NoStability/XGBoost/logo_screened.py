import argparse
import itertools
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
import matplotlib.pyplot as plt
from xgboost import XGBRegressor

# Parse command-line argument for test cluster (LOGO setting)
parser = argparse.ArgumentParser(description="LOGO Stabilized Regression using XGBoost with Feature Importance Screening")
parser.add_argument("--test_cluster", type=int, required=True, help="Cluster to leave out for testing")
args = parser.parse_args()

# Read and preprocess data
df = pd.read_csv('/cluster/project/math/akmete/MSc/preprocessing/df_balanced_groups_onevegindex.csv')
df = df.drop(columns=['Unnamed: 0'])  # Drop unnecessary columns
df = df.dropna(axis=1, how='all')       # Drop columns where all values are NaN
df = df.fillna(0)
print("Columns:", df.columns)
print("Loaded dataframe.")

# Convert float64 to float32
for col in tqdm(df.select_dtypes(include=['float64']).columns, desc="Casting columns"):
    df[col] = df[col].astype('float32')

# Define feature and target columns
initial_feature_columns = [col for col in df.columns if col not in ['GPP', 'site_id', 'cluster']]
target_column = "GPP"

# Split data into training and testing
test_cluster = args.test_cluster
if test_cluster not in df['cluster'].unique():
    raise ValueError("Invalid test cluster value.")
df_train = df[df['cluster'] != test_cluster].copy()
df_test  = df[df['cluster'] == test_cluster].copy()


# SCREENING
X_train_screen = df_train[initial_feature_columns]
y_train_screen = df_train[target_column]
scaler_screen = MinMaxScaler()
X_train_screen_scaled = scaler_screen.fit_transform(X_train_screen)

temp_model = XGBRegressor(objective='reg:squarederror',
                          n_estimators=50,
                          max_depth=3,
                          learning_rate=0.1,
                          random_state=42,
                          n_jobs=-1)
temp_model.fit(X_train_screen_scaled, y_train_screen)
importance = temp_model.feature_importances_
feature_importance = dict(zip(initial_feature_columns, importance))
sorted_features = sorted(feature_importance, key=feature_importance.get, reverse=True)
top_k = 7
selected_features = sorted_features[:top_k]
print("Selected features after XGBoost screening (top 7):", selected_features)

# Screen down the feature set given feature importances
feature_columns = list(selected_features)
if not feature_columns:
    raise ValueError("Feature screening removed all features. Please check your data or adjust screening parameters.")

# Create all subset of features
all_subsets = []
for r in range(1, len(feature_columns) + 1):
    for subset in itertools.combinations(feature_columns, r):
        all_subsets.append(list(subset))
print("Number of subsets:", len(all_subsets))

# For a given subset S, train an XGBoost regressor on all training data and compute the prediction score.
def compute_logo_pred_score(df_train, feature_subset, target_column):
    # Train on all training data using features in subset S
    X_train_subset = df_train[feature_subset]
    y_train_subset = df_train[target_column]
    
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train_subset)
    
    # Scale y using global training y parameters
    y_train_min = y_train_subset.min()
    y_train_max = y_train_subset.max()
    if y_train_max - y_train_min == 0:
        y_train_scaled = y_train_subset.values
    else:
        y_train_scaled = (y_train_subset - y_train_min) / (y_train_max - y_train_min)
    
    # Use XGBoost regressor
    model = XGBRegressor(n_estimators=100, learning_rate=0.1, reg_alpha=1, reg_lambda=1,
                         random_state=42, objective='reg:squarederror')
    model.fit(X_train_scaled, y_train_scaled)
    
    # Compute MSE for each training site using the trained regressor
    training_sites = df_train['site_id'].unique()
    mse_list = []
    for site in training_sites:
        df_site = df_train[df_train['site_id'] == site]
        X_site = df_site[feature_subset]
        y_site = df_site[target_column]
        
        # Use the same scaler fitted on the entire training data
        X_site_scaled = scaler.transform(X_site)
        
        # scale the target given training statistics
        if y_train_max - y_train_min == 0:
            y_site_scaled = y_site.values
        else:
            y_site_scaled = (y_site - y_train_min) / (y_train_max - y_train_min)
        
        # calculate metrics for prediction score
        y_pred = model.predict(X_site_scaled)
        mse = mean_squared_error(y_site_scaled, y_pred)
        mse_list.append(mse)
    
    avg_mse = np.mean(mse_list)
    return -avg_mse  # Negative average MSE as prediction score

# for all subsets, calculate prediction score
pred_scores_all = []
for subset in tqdm(all_subsets, desc="Evaluating subsets (LOGO) on training sites"):
    score = compute_logo_pred_score(df_train, subset, target_column)
    pred_scores_all.append(score)

# Set alpha_pred and filter to get predictive subsets
alpha_pred = 0.1
c_pred = np.quantile(pred_scores_all, 1 - alpha_pred)
O_hat = [subset for subset, score in zip(all_subsets, pred_scores_all) if score >= c_pred]
print("For test cluster", test_cluster, "O_hat count:", len(O_hat))

# Train the ensemble model (ie stabilized regressor)
trained_models = {}
for subset in O_hat:
    X_train_subset = df_train[subset]
    y_train_subset = df_train[target_column]
    
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train_subset)
    
    y_train_min = y_train_subset.min()
    y_train_max = y_train_subset.max()
    if y_train_max - y_train_min == 0:
        y_train_scaled = y_train_subset.values
    else:
        y_train_scaled = (y_train_subset - y_train_min) / (y_train_max - y_train_min)
    
    model = XGBRegressor(n_estimators=100, learning_rate=0.1, reg_alpha=1, reg_lambda=1,
                         random_state=42, objective='reg:squarederror')
    model.fit(X_train_scaled, y_train_scaled)
    trained_models[str(subset)] = (model, scaler, y_train_min, y_train_max)

# each regressor gets the same weight
weight = 1.0 / len(O_hat) if len(O_hat) > 0 else 0

# Define prediction function of the stabilized regressor
def ensemble_predict(X, return_scaled=False):
    ensemble_preds = np.zeros(len(X))
    for subset in O_hat:
        model, scaler, y_min, y_max = trained_models[str(subset)]
        X_subset = X[subset]
        X_subset_scaled = scaler.transform(X_subset)
        pred_scaled = model.predict(X_subset_scaled)
        if return_scaled:
            pred = pred_scaled
        else:
            # Convert predictions back to the original target space
            pred = pred_scaled if (y_max - y_min) == 0 else pred_scaled * (y_max - y_min) + y_min
        ensemble_preds += weight * pred
    return ensemble_preds

# Train a full model corresponding to training on all screened features
X_train_full = df_train[feature_columns]
y_train_full = df_train[target_column]
scaler_full = MinMaxScaler()
X_train_full_scaled = scaler_full.fit_transform(X_train_full)
y_train_min_full = y_train_full.min()
y_train_max_full = y_train_full.max()
if y_train_max_full - y_train_min_full == 0:
    y_train_full_scaled = y_train_full.values
else:
    y_train_full_scaled = (y_train_full - y_train_min_full) / (y_train_max_full - y_train_min_full)

full_model = XGBRegressor(n_estimators=100, learning_rate=0.1, reg_alpha=1, reg_lambda=1,
                          random_state=42, objective='reg:squarederror')
full_model.fit(X_train_full_scaled, y_train_full_scaled)

# Evaluate full model and ensemble model
X_test_full = df_test[feature_columns]
X_test_full_scaled = scaler_full.transform(X_test_full)
full_preds_scaled = full_model.predict(X_test_full_scaled)
if y_train_max_full - y_train_min_full == 0:
    y_test_scaled_full = df_test[target_column].values
else:
    y_test_scaled_full = (df_test[target_column].values - y_train_min_full) / (y_train_max_full - y_train_min_full)

full_mse_scaled = mean_squared_error(y_test_scaled_full, full_preds_scaled)
full_rmse_scaled = np.sqrt(full_mse_scaled)
full_r2 = r2_score(y_test_scaled_full, full_preds_scaled)
full_relative_error = np.mean(np.abs(y_test_scaled_full - full_preds_scaled) / np.abs(y_test_scaled_full))
full_mae = np.mean(np.abs(y_test_scaled_full - full_preds_scaled))

ensemble_preds_scaled = ensemble_predict(df_test, return_scaled=True)
ensemble_mse_scaled = mean_squared_error(y_test_scaled_full, ensemble_preds_scaled)
ensemble_rmse_scaled = np.sqrt(ensemble_mse_scaled)
ensemble_r2 = r2_score(y_test_scaled_full, ensemble_preds_scaled)
ensemble_relative_error = np.mean(np.abs(y_test_scaled_full - ensemble_preds_scaled) / np.abs(y_test_scaled_full))
ensemble_mae = np.mean(np.abs(y_test_scaled_full - ensemble_preds_scaled))

print("Test Cluster {} Ensemble MSE (scaled): {}".format(test_cluster, ensemble_mse_scaled))
print("Test Cluster {} Full model MSE (scaled): {}".format(test_cluster, full_mse_scaled))

# Save results
results = {
    "test_cluster": test_cluster,
    "ensemble_mse_scaled": ensemble_mse_scaled,
    "full_mse_scaled": full_mse_scaled,
    "ensemble_rmse_scaled": ensemble_rmse_scaled,
    "full_rmse_scaled": full_rmse_scaled,
    "ensemble_r2": ensemble_r2,
    "full_r2": full_r2,
    "ensemble_relative_error": ensemble_relative_error,
    "full_relative_error": full_relative_error,
    "ensemble_mae": ensemble_mae,
    "full_mae": full_mae,
    "O_hat_count": len(O_hat),
    "O_hat": O_hat
}
# Save to csv
results_df = pd.DataFrame([results])
results_df.to_csv(f'results_cluster_{test_cluster}.csv', index=False)
print("Results saved to results_cluster_{}.csv".format(test_cluster))

# Plot distribution of the prediction scores if needed
plt.figure()
plt.hist(pred_scores_all, bins=50)
plt.title("Prediction Scores (Training Sites, Cluster {} left out)".format(test_cluster))
plt.xlabel("Prediction Score")
plt.ylabel("Frequency")
plt.savefig(f'pred_scores_cluster_{test_cluster}.png')
plt.close()
