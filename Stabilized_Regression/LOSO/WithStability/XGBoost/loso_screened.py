# Import libraries
import sys
import itertools
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# Command-line argument handling for job array
if len(sys.argv) < 2:
    print("Usage: python loso_array_xgb_with_stability.py <test_site_index>")
    sys.exit(1)
test_site_index = int(sys.argv[1])

# Load and process data
df = pd.read_csv('/cluster/project/math/akmete/MSc/50sites_balanced.csv')
df = df.dropna(axis=1, how='all') # Remove all columns with only NaNS
df = df.fillna(0) # Fill NaN with 0 (if there are any
df = df.drop(columns=['Unnamed: 0', 'cluster']) # drop unnecessary columns
print("Loaded dataframe with columns:", df.columns)

# Convert float64 to float32
for col in df.select_dtypes(include=['float64']).columns:
    df[col] = df[col].astype('float32')

# Define feature and target column
feature_columns = [col for col in df.columns if col not in ['GPP', 'site_id']]
target_column = "GPP"

# Get all unique sites from the dataframe.
sites = df['site_id'].unique()

if test_site_index < 0 or test_site_index >= len(sites):
    print(f"Error: test_site_index {test_site_index} is out of range (0 to {len(sites)-1}).")
    sys.exit(1)
test_site = sites[test_site_index]
print(f"Processing fold (test site): {test_site}")

# SCREENING on training data (use not too complex trees in XGBoost as it is used only for screening)
df_train = df[df['site_id'] != test_site].copy()
df_test  = df[df['site_id'] == test_site].copy()

X_train_initial = df_train[feature_columns]
y_train_initial = df_train[target_column]
scaler_screen = MinMaxScaler()
X_train_initial_scaled = scaler_screen.fit_transform(X_train_initial)

screening_model = xgb.XGBRegressor(random_state=0, n_jobs=-1)
screening_model.fit(X_train_initial_scaled, y_train_initial)
importances = screening_model.feature_importances_
sorted_indices = np.argsort(importances)[::-1]
# Select top 7 features based on importance.
selected_features = [feature_columns[i] for i in sorted_indices[:7]]
if not selected_features:
    raise ValueError(f"XGBoost screening removed all features for test site {test_site}.")
print(f"Selected top 7 features for test site {test_site}: {selected_features}")

# Given screened subsets of features, get all possible combinations of features
all_subsets = []
for r in range(1, len(selected_features) + 1):
    for subset in itertools.combinations(selected_features, r):
        all_subsets.append(list(subset))
print("Number of subsets:", len(all_subsets))

# Define function computing scores
def compute_scores(df_train, feature_subset, target_column):
    X_train = df_train[feature_subset]
    y_train = df_train[target_column]
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    y_train_min = y_train.min()
    y_train_max = y_train.max()
    if y_train_max - y_train_min == 0:
        y_train_scaled = y_train.values
    else:
        y_train_scaled = (y_train - y_train_min) / (y_train_max - y_train_min)
    
    model = xgb.XGBRegressor(random_state=0, n_jobs=-1)
    model.fit(X_train_scaled, y_train_scaled)
    
    # Compute per-site MSEs.
    training_sites = df_train['site_id'].unique()
    mse_list = []
    for site in training_sites:
        df_site = df_train[df_train['site_id'] == site]
        X_site = df_site[feature_subset]
        y_site = df_site[target_column]
        X_site_scaled = scaler.transform(X_site)
        if y_train_max - y_train_min == 0:
            y_site_scaled = y_site.values
        else:
            y_site_scaled = (y_site - y_train_min) / (y_train_max - y_train_min)
        y_pred = model.predict(X_site_scaled)
        mse = mean_squared_error(y_site_scaled, y_pred)
        mse_list.append(mse)
    
    avg_mse = np.mean(mse_list)
    pred_score = -avg_mse  # Higher (less negative) is better.
    stability_score = np.quantile(mse_list, 0.95)
    return pred_score, stability_score, model, scaler, y_train_min, y_train_max

# Evaluate each subset, that is get for each subset the prediction score and stability score and save them
pred_scores_all = []
stability_scores_all = []
scores_info = {}  # to store additional info per subset
for subset in all_subsets:
    pred_score, stab_score, model, scaler, y_min, y_max = compute_scores(df_train, subset, target_column)
    pred_scores_all.append(pred_score)
    stability_scores_all.append(stab_score)
    scores_info[str(subset)] = {
        "pred_score": pred_score,
        "stab_score": stab_score,
        "model": model,
        "scaler": scaler,
        "y_min": y_min,
        "y_max": y_max
    }

# Given subsets S and score function, we filter first with respect to stability
quantile_level = 0.1  # e.g., 10th quantile threshold for stability
alpha_stab = np.quantile(stability_scores_all, quantile_level)
G_hat = [subset for subset, stab in zip(all_subsets, stability_scores_all) if stab <= alpha_stab]
print(f"G_hat count for test site {test_site}: {len(G_hat)}")

# From the stable subsets, we only keep the predictive sets (this set we call O_hat)
alpha_pred = 0.05
g_hat_pred_scores = [scores_info[str(subset)]["pred_score"] for subset in G_hat]
c_pred = np.quantile(g_hat_pred_scores, 1 - alpha_pred)
O_hat = [subset for subset in G_hat if scores_info[str(subset)]["pred_score"] >= c_pred]
print(f"O_hat count for test site {test_site}: {len(O_hat)}")

# Train the ensemble model
trained_models = {}
for subset in O_hat:
    X_train_subset = df_train[subset]
    y_train_subset = df_train[target_column]
    scaler_model = MinMaxScaler()
    X_train_scaled = scaler_model.fit_transform(X_train_subset)
    y_train_min = y_train_subset.min()
    y_train_max = y_train_subset.max()
    if y_train_max - y_train_min == 0:
        y_train_scaled = y_train_subset.values
    else:
        y_train_scaled = (y_train_subset - y_train_min) / (y_train_max - y_train_min)
    model = xgb.XGBRegressor(random_state=0, n_jobs=-1)
    model.fit(X_train_scaled, y_train_scaled)
    trained_models[str(subset)] = (model, scaler_model, y_train_min, y_train_max)

# Define weight define prediction function (ie define how SR is defined)
weight = 1.0 / len(O_hat) if len(O_hat) > 0 else 0
def ensemble_predict(X, return_scaled=False):
    ensemble_preds = np.zeros(len(X))
    for subset in O_hat:
        model, scaler_model, y_min, y_max = trained_models[str(subset)]
        X_subset = X[subset]
        X_subset_scaled = scaler_model.transform(X_subset)
        pred_scaled = model.predict(X_subset_scaled)
        if return_scaled:
            pred = pred_scaled
        else:
            pred = pred_scaled if (y_max - y_min) == 0 else pred_scaled * (y_max - y_min) + y_min
        ensemble_preds += weight * pred
    return ensemble_preds

# Train full model on screened featuers
X_train_full = df_train[selected_features]
y_train_full = df_train[target_column]
scaler_full = MinMaxScaler()
X_train_full_scaled = scaler_full.fit_transform(X_train_full)
y_train_min_full = y_train_full.min()
y_train_max_full = y_train_full.max()
if y_train_max_full - y_train_min_full == 0:
    y_train_full_scaled = y_train_full.values
else:
    y_train_full_scaled = (y_train_full - y_train_min_full) / (y_train_max_full - y_train_min_full)
full_model = xgb.XGBRegressor(random_state=0, n_jobs=-1)
full_model.fit(X_train_full_scaled, y_train_full_scaled)

# Evaluate both models on the test site and get metrics from both models
X_test_full = df_test[selected_features]
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

print(f"Test Site {test_site} Ensemble MSE (scaled): {ensemble_mse_scaled}")
print(f"Test Site {test_site} Full model MSE (scaled): {full_mse_scaled}")

# To compare prediction and stability scores from different sites, save them
plt.figure()
plt.hist(pred_scores_all, bins=50)
plt.title(f"Prediction Scores (Fold: Site {test_site} left out)")
plt.xlabel("Prediction Score")
plt.ylabel("Frequency")
plt.savefig(f'pred_scores_site_{test_site}.png')
plt.close()

plt.figure()
plt.hist(stability_scores_all, bins=50)
plt.title(f"Stability Scores (Fold: Site {test_site} left out)")
plt.xlabel("Stability Score")
plt.ylabel("Frequency")
plt.savefig(f'stability_scores_site_{test_site}.png')
plt.close()

# --- Save results for this fold ---
fold_result = {
    "test_site": test_site,
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
    "G_hat_count": len(G_hat),
    "O_hat_count": len(O_hat),
    "O_hat": O_hat
}
results_df = pd.DataFrame([fold_result])
results_df.to_csv(f'results_site_{test_site}.csv', index=False)
print(f"Results saved to results_site_{test_site}.csv")

