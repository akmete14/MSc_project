import argparse
import itertools
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
import matplotlib.pyplot as plt

# Set random state for reproducibility
random_state = 42

# Parse command-line argument for site index (for job array parallelism)
parser = argparse.ArgumentParser(description="Insite Stabilized Regression with Feature Screening")
parser.add_argument("--site_index", type=int, required=True, help="Index of the site to process")
args = parser.parse_args()

# Read data and preprocess
df = pd.read_csv('/cluster/project/math/akmete/MSc/preprocessing/df_balanced_groups_onevegindex.csv')
df = df.dropna(axis=1, how='all')  # Drop columns where all values are NaN
df = df.fillna(0)
for col in tqdm(df.select_dtypes(include=['float64']).columns, desc="Casting columns"):
    df[col] = df[col].astype('float32')
df = df.drop(columns=['Unnamed: 0', 'cluster'])
print("Columns:", df.columns)
print("Loaded dataframe.")

# Define all possible features and the target
all_features = [col for col in df.columns if col not in ['GPP', 'site_id']]
target_column = "GPP"

# -------------------------------
# Feature Importance Screening
# -------------------------------
# Fit a preliminary model on the entire dataset to compute feature importances.
X_all = df[all_features]
y_all = df[target_column]
scaler_pre = MinMaxScaler()
X_all_scaled = scaler_pre.fit_transform(X_all)

temp_model = XGBRegressor(objective='reg:squarederror',
                          n_estimators=50,
                          max_depth=3,
                          learning_rate=0.1,
                          random_state=random_state,
                          n_jobs=-1)
temp_model.fit(X_all_scaled, y_all)

# Get feature importances and select the top K features (e.g., top 10)
importance = temp_model.feature_importances_
feature_importance = dict(zip(all_features, importance))
sorted_features = sorted(feature_importance, key=feature_importance.get, reverse=True)
top_k = 7
screened_features = sorted_features[:top_k]
print("Screened features (top {}): {}".format(top_k, screened_features))

# Create all feature subsets from the screened features
all_subsets = []
for r in range(1, len(screened_features) + 1):
    for subset in itertools.combinations(screened_features, r):
        all_subsets.append(list(subset))
print("Number of subsets:", len(all_subsets))

# Get unique sites and select the site to process from the command-line argument
sites = sorted(df['site_id'].unique())
if args.site_index < 0 or args.site_index >= len(sites):
    raise ValueError(f"Invalid site index. Must be between 0 and {len(sites)-1}.")
site = sites[args.site_index]
print("Processing site:", site)
df_site = df[df['site_id'] == site].copy()

# Define a compute function for a single site using an 80/20 chronological split
def compute_scores_insite(df_site, feature_subset, target_column):
    split_index = int(0.8 * len(df_site))
    df_train = df_site.iloc[:split_index]
    df_test  = df_site.iloc[split_index:]
    
    X_train = df_train[feature_subset]
    y_train = df_train[target_column]
    X_test  = df_test[feature_subset]
    y_test  = df_test[target_column]
    
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)
    
    y_train_min = y_train.min()
    y_train_max = y_train.max()
    if y_train_max - y_train_min == 0:
        y_train_scaled = y_train.values
    else:
        y_train_scaled = (y_train - y_train_min) / (y_train_max - y_train_min)
    
    model = XGBRegressor(objective='reg:squarederror',
                         n_estimators=100,
                         max_depth=5,
                         learning_rate=0.1,
                         random_state=random_state,
                         n_jobs=-1)
    model.fit(X_train_scaled, y_train_scaled)
    
    # Compute prediction score as negative MSE on training data
    y_pred_train = model.predict(X_train_scaled)
    mse_train = mean_squared_error(y_train_scaled, y_pred_train)
    pred_score = -mse_train
    
    # Stability score: variance of squared errors on training data
    squared_errors = (y_train_scaled - y_pred_train) ** 2
    stab_score = np.var(squared_errors, ddof=1) if len(squared_errors) > 1 else 0
    
    return pred_score, stab_score, model, scaler, y_train_min, y_train_max

# Compute scores for each subset on the current site
pred_scores_all = []
stab_scores_all = []

for subset in tqdm(all_subsets, desc=f"Evaluating subsets for site {site}"):
    pred_score, stab_score, _, _, _, _ = compute_scores_insite(df_site, subset, target_column)
    pred_scores_all.append(pred_score)
    stab_scores_all.append(stab_score)

# Set threshold based on prediction scores (e.g., top 5%)
alpha_pred = 0.05  
c_pred = np.quantile(pred_scores_all, 1 - alpha_pred)
O_hat = [subset for subset, score in zip(all_subsets, pred_scores_all) if score >= c_pred]
print("For site", site, "O_hat count:", len(O_hat))

# Perform an 80/20 chronological split on this site
split_index = int(0.8 * len(df_site))
df_train_site = df_site.iloc[:split_index]
df_test_site  = df_site.iloc[split_index:]

# Train ensemble models for each subset in O_hat on the training portion
trained_models = {}
for subset in O_hat:
    X_train = df_train_site[subset]
    y_train = df_train_site[target_column]
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    y_train_min = y_train.min()
    y_train_max = y_train.max()
    if y_train_max - y_train_min == 0:
        y_train_scaled = y_train.values
    else:
        y_train_scaled = (y_train - y_train_min) / (y_train_max - y_train_min)
    
    model = XGBRegressor(objective='reg:squarederror',
                         n_estimators=100,
                         max_depth=5,
                         learning_rate=0.1,
                         random_state=random_state,
                         n_jobs=-1)
    model.fit(X_train_scaled, y_train_scaled)
    trained_models[str(subset)] = (model, scaler, y_train_min, y_train_max)

weight = 1.0 / len(O_hat) if len(O_hat) > 0 else 0

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
            pred = pred_scaled if (y_max - y_min) == 0 else pred_scaled * (y_max - y_min) + y_min
        ensemble_preds += weight * pred
    return ensemble_preds

# Train a full model using all screened features on the training portion for comparison
X_train_full = df_train_site[screened_features]
y_train_full = df_train_site[target_column]
scaler_full = MinMaxScaler()
X_train_full_scaled = scaler_full.fit_transform(X_train_full)
y_train_min_full = y_train_full.min()
y_train_max_full = y_train_full.max()
if y_train_max_full - y_train_min_full == 0:
    y_train_full_scaled = y_train_full.values
else:
    y_train_full_scaled = (y_train_full - y_train_min_full) / (y_train_max_full - y_train_min_full)

full_model = XGBRegressor(objective='reg:squarederror',
                          n_estimators=100,
                          max_depth=5,
                          learning_rate=0.1,
                          random_state=random_state,
                          n_jobs=-1)
full_model.fit(X_train_full_scaled, y_train_full_scaled)

X_test_full = df_test_site[screened_features]
X_test_full_scaled = scaler_full.transform(X_test_full)
full_preds_scaled = full_model.predict(X_test_full_scaled)
if y_train_max_full - y_train_min_full == 0:
    y_test_scaled_full = df_test_site[target_column].values
else:
    y_test_scaled_full = (df_test_site[target_column].values - y_train_min_full) / (y_train_max_full - y_train_min_full)
full_mse_scaled = mean_squared_error(y_test_scaled_full, full_preds_scaled)
full_rmse_scaled = np.sqrt(full_mse_scaled)
full_r2 = r2_score(y_test_scaled_full, full_preds_scaled)
full_relative_error = np.mean(np.abs(y_test_scaled_full - full_preds_scaled) / np.abs(y_test_scaled_full))
full_mae = np.mean(np.abs(y_test_scaled_full - full_preds_scaled))

# Compute ensemble predictions in scaled space
ensemble_preds_scaled = ensemble_predict(df_test_site, return_scaled=True)
ensemble_mse_scaled = mean_squared_error(y_test_scaled_full, ensemble_preds_scaled)
ensemble_rmse_scaled = np.sqrt(ensemble_mse_scaled)
ensemble_r2 = r2_score(y_test_scaled_full, ensemble_preds_scaled)
ensemble_relative_error = np.mean(np.abs(y_test_scaled_full - ensemble_preds_scaled) / np.abs(y_test_scaled_full))
ensemble_mae = np.mean(np.abs(y_test_scaled_full - ensemble_preds_scaled))

print("Site {} Ensemble MSE (scaled): {}".format(site, ensemble_mse_scaled))
print("Site {} Full model MSE (scaled): {}".format(site, full_mse_scaled))

# Save results
results = {
    "site": site,
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

results_df = pd.DataFrame([results])
results_df.to_csv(f'results_site_{site}.csv', index=False)
print("Results saved to results_site_{}.csv".format(site))

# For every 10th site, generate and save plots of the score distributions.
if args.site_index % 10 == 0:
    plt.figure()
    plt.hist(stab_scores_all, bins=50)
    plt.title("Stability Scores (Site: {})".format(site))
    plt.xlabel("Stability Score")
    plt.ylabel("Frequency")
    plt.savefig(f'stab_scores_{site}.png')
    plt.close()

    plt.figure()
    plt.hist(pred_scores_all, bins=50)
    plt.title("Prediction Scores (Site: {})".format(site))
    plt.xlabel("Prediction Score")
    plt.ylabel("Frequency")
    plt.savefig(f'pred_scores_{site}.png')
    plt.close()
