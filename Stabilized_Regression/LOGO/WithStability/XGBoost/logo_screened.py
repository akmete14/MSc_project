import argparse
import itertools
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
import matplotlib.pyplot as plt
from xgboost import XGBRegressor

# -----------------------------
# Parse command-line argument for test cluster (LOGO setting)
# -----------------------------
parser = argparse.ArgumentParser(description="LOGO Stabilized Regression using XGBoost with Dual Filtering")
parser.add_argument("--test_cluster", type=int, required=True, help="Cluster to leave out for testing")
args = parser.parse_args()

# -----------------------------
# Load and preprocess data
# -----------------------------
df = pd.read_csv('/cluster/project/math/akmete/MSc/preprocessing/df_balanced_groups_onevegindex.csv')
df = df.drop(columns=['Unnamed: 0'])  # Drop unnecessary columns
df = df.dropna(axis=1, how='all')       # Drop columns where all values are NaN
df = df.fillna(0)

# Cast float columns to float32 for efficiency
for col in df.select_dtypes(include=['float64']).columns:
    df[col] = df[col].astype('float32')

print("Columns:", df.columns)
print("Loaded dataframe.")

# -----------------------------
# Define initial features and target.
# Exclude 'GPP', 'site_id', and 'cluster'
# -----------------------------
initial_feature_columns = [col for col in df.columns if col not in ['GPP', 'site_id', 'cluster']]
target_column = "GPP"

# -----------------------------
# Split data into training (all clusters except the test cluster) and testing (held-out cluster)
# -----------------------------
test_cluster = args.test_cluster
if test_cluster not in df['cluster'].unique():
    raise ValueError("Invalid test cluster value.")
df_train = df[df['cluster'] != test_cluster].copy()
df_test  = df[df['cluster'] == test_cluster].copy()

# -----------------------------
# FEATURE SCREENING USING XGBOOST
# Compute feature importances on the training set and select the top 5 features.
# -----------------------------
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
top_k = 6
selected_features = sorted_features[:top_k]
print("Selected features after screening (top 6):", selected_features)

# Update feature_columns to only include selected features.
feature_columns = list(selected_features)
if not feature_columns:
    raise ValueError("Feature screening removed all features.")

# -----------------------------
# Generate all nonempty feature subsets from the screened features.
# -----------------------------
all_subsets = []
for r in range(1, len(feature_columns) + 1):
    for subset in itertools.combinations(feature_columns, r):
        all_subsets.append(list(subset))
print("Number of subsets:", len(all_subsets))

# -----------------------------
# Define function to compute prediction and instability scores for a subset using XGBoost.
#
# For a given subset S, train an XGBRegressor on the entire training data (all non-held-out clusters).
# Then, for each training site (from 'site_id'), compute the MSE after scaling (using global training scaling).
#
# Returns:
#   - pred_score: negative average MSE over training sites.
#   - instability_score: the 95th quantile of per-site MSE values.
#   - Plus the fitted model, scaler, and y_train scaling parameters.
# -----------------------------
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

    model = XGBRegressor(n_estimators=100, learning_rate=0.1, reg_alpha=1, reg_lambda=1,
                         random_state=42, objective='reg:squarederror')
    model.fit(X_train_scaled, y_train_scaled)

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
    instability_score = np.quantile(mse_list, 0.95)
    return pred_score, instability_score, model, scaler, y_train_min, y_train_max

# -----------------------------
# Evaluate every subset on the training data.
# -----------------------------
pred_scores_all = []
instability_scores_all = []
scores_info = {}  # Dictionary to hold scores and model info for each subset.
for subset in tqdm(all_subsets, desc=f"Evaluating subsets for cluster {test_cluster}"):
    pred_score, instab_score, model, scaler, y_min, y_max = compute_scores(df_train, subset, target_column)
    pred_scores_all.append(pred_score)
    instability_scores_all.append(instab_score)
    scores_info[str(subset)] = {"pred_score": pred_score,
                                "instab_score": instab_score,
                                "model": model,
                                "scaler": scaler,
                                "y_min": y_min,
                                "y_max": y_max}

# -----------------------------
# Filter subsets using dual criteria:
#   1. First, select subsets (G_hat) with instability score <= the 5th quantile.
#   2. Then, among G_hat, select subsets (O_hat) whose prediction score is in the top 5%.
# -----------------------------
instab_threshold = np.quantile(instability_scores_all, 0.05)
G_hat = [subset for subset, instab in zip(all_subsets, instability_scores_all) if instab <= instab_threshold]
print("For test cluster", test_cluster, "G_hat count:", len(G_hat))

# Compute the prediction scores only for subsets in G_hat.
ghat_pred_scores = [score for subset, score in zip(all_subsets, pred_scores_all) if subset in G_hat]
alpha_pred = 0.1  # keep the top 10% based on prediction score
c_pred = np.quantile(ghat_pred_scores, 1 - alpha_pred)

# Now, select O_hat as the subsets in G_hat whose prediction score is >= c_pred.
O_hat = [subset for subset, score in zip(all_subsets, pred_scores_all) if (subset in G_hat) and (score >= c_pred)]
print("For test cluster", test_cluster, "O_hat count:", len(O_hat))
print("O_hat subsets for test cluster", test_cluster, ":", O_hat)


# -----------------------------
# Train ensemble models for each subset in O_hat using the entire training data.
# -----------------------------
trained_models = {}
for subset in O_hat:
    X_train_subset = df_train[subset]
    y_train_subset = df_train[target_column]
    scaler_subset = MinMaxScaler()
    X_train_scaled = scaler_subset.fit_transform(X_train_subset)
    y_train_min = y_train_subset.min()
    y_train_max = y_train_subset.max()
    if y_train_max - y_train_min == 0:
        y_train_scaled = y_train_subset.values
    else:
        y_train_scaled = (y_train_subset - y_train_min) / (y_train_max - y_train_min)
    model = XGBRegressor(n_estimators=100, learning_rate=0.1, reg_alpha=1, reg_lambda=1,
                         random_state=42, objective='reg:squarederror')
    model.fit(X_train_scaled, y_train_scaled)
    trained_models[str(subset)] = (model, scaler_subset, y_train_min, y_train_max)

# Use equal weight for each model in the ensemble.
weight = 1.0 / len(O_hat) if len(O_hat) > 0 else 0

def ensemble_predict(X, return_scaled=False):
    ensemble_preds = np.zeros(len(X))
    for subset in O_hat:
        model, scaler_subset, y_min, y_max = trained_models[str(subset)]
        X_subset = X[subset]
        X_subset_scaled = scaler_subset.transform(X_subset)
        pred_scaled = model.predict(X_subset_scaled)
        if return_scaled:
            pred = pred_scaled
        else:
            pred = pred_scaled if (y_max - y_min) == 0 else pred_scaled * (y_max - y_min) + y_min
        ensemble_preds += weight * pred
    return ensemble_preds

# -----------------------------
# Train a full model using all screened features (i.e. the top 5 features) on the entire training set.
# -----------------------------
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

# -----------------------------
# Evaluate both the ensemble model and the full model on the held-out (test) cluster.
# -----------------------------
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

# -----------------------------
# Save results
# -----------------------------
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
    "G_hat_count": len(G_hat),
    "O_hat_count": len(O_hat),
    "O_hat": O_hat
}

results_df = pd.DataFrame([results])
results_df.to_csv(f'results_cluster_{test_cluster}.csv', index=False)
print("Results saved to results_cluster_{}.csv".format(test_cluster))

# -----------------------------
# Save diagnostic plots.
# -----------------------------
plt.figure()
plt.hist(pred_scores_all, bins=50)
plt.title("Prediction Scores (Training Sites, Cluster {} left out)".format(test_cluster))
plt.xlabel("Prediction Score")
plt.ylabel("Frequency")
plt.savefig(f'pred_scores_cluster_{test_cluster}.png')
plt.close()

plt.figure()
plt.hist(instability_scores_all, bins=50)
plt.title("Instability Scores (Training Sites, Cluster {} left out)".format(test_cluster))
plt.xlabel("Instability Score")
plt.ylabel("Frequency")
plt.savefig(f'instability_scores_cluster_{test_cluster}.png')
plt.close()
