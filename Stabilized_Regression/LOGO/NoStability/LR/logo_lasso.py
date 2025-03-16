# Import libraries
import argparse
import itertools
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, LassoCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
import matplotlib.pyplot as plt

# Parse command-line argument for test cluster (LOGO setting)
parser = argparse.ArgumentParser(description="LOGO Stabilized Regression using Linear Regression with LASSO feature screening")
parser.add_argument("--test_cluster", type=int, required=True, help="Cluster to leave out for testing")
args = parser.parse_args()

# Load and process data
df = pd.read_csv('/cluster/project/math/akmete/MSc/preprocessing/df_balanced_groups_onevegindex.csv')
df = df.drop(columns=['Unnamed: 0'])  # Drop unnecessary columns
df = df.dropna(axis=1, how='all')  # Drop columns where all values are NaN
df = df.fillna(0)

# Covnert float64 to float32 ro save resources
for col in tqdm(df.select_dtypes(include=['float64']).columns, desc="Casting columns"):
    df[col] = df[col].astype('float32')

print("Columns:", df.columns)
print("Loaded dataframe.")

# Define feature and target columns
initial_feature_columns = [col for col in df.columns if col not in ['GPP', 'site_id', 'cluster']]
target_column = "GPP"

# Split data into training and test
test_cluster = args.test_cluster
if test_cluster not in df['cluster'].unique():
    raise ValueError("Invalid test cluster value.")
df_train = df[df['cluster'] != test_cluster].copy()
df_test  = df[df['cluster'] == test_cluster].copy()

# SCREENING
X_train = df_train[initial_feature_columns]
y_train = df_train[target_column]

# Scale features (LASSO is sensitive to feature scales)
scaler_lasso = MinMaxScaler()
X_train_scaled = scaler_lasso.fit_transform(X_train)

# Use LassoCV to select features via cross-validation
lasso = LassoCV(cv=5, random_state=0)
lasso.fit(X_train_scaled, y_train)

# Identify features with non-zero coefficients
selected_features = np.array(initial_feature_columns)[lasso.coef_ != 0]
print("Selected features after LASSO screening:", selected_features)

# Update feature_columns to only include selected features.
feature_columns = list(selected_features)
if not feature_columns:
    raise ValueError("LASSO screening removed all features. Please check your data or adjust LASSO parameters.")



# Create set of all possible subsets from the screened variables
all_subsets = []
for r in range(1, len(feature_columns) + 1):
    for subset in itertools.combinations(feature_columns, r):
        all_subsets.append(list(subset))
print("Number of subsets:", len(all_subsets))

# For a given subset S, train a regressor on all training data and compute the prediction score.
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
    
    model = LinearRegression()
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
        
        # Scale the site's y-values using the global training y parameters
        if y_train_max - y_train_min == 0:
            y_site_scaled = y_site.values
        else:
            y_site_scaled = (y_site - y_train_min) / (y_train_max - y_train_min)
        
        y_pred = model.predict(X_site_scaled)
        mse = mean_squared_error(y_site_scaled, y_pred)
        mse_list.append(mse)
    
    avg_mse = np.mean(mse_list)
    return -avg_mse  # Negative average MSE as prediction score

# For each set of features, calculate the corresponding pred score
pred_scores_all = []
for subset in tqdm(all_subsets, desc="Evaluating subsets (LOGO) on training sites"):
    score = compute_logo_pred_score(df_train, subset, target_column)
    pred_scores_all.append(score)

# Filter only subsets with prediction score above threshold
alpha_pred = 0.05
c_pred = np.quantile(pred_scores_all, 1 - alpha_pred)
O_hat = [subset for subset, score in zip(all_subsets, pred_scores_all) if score >= c_pred]
print("For test cluster", test_cluster, "O_hat count:", len(O_hat))

# Given O_hat, train the ensmeble model
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
    
    model = LinearRegression()
    model.fit(X_train_scaled, y_train_scaled)
    trained_models[str(subset)] = (model, scaler, y_train_min, y_train_max)

# Every regressor gets same weight
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
            # Convert predictions back to the original target space
            pred = pred_scaled if (y_max - y_min) == 0 else pred_scaled * (y_max - y_min) + y_min
        ensemble_preds += weight * pred
    return ensemble_preds

# Train full model (corresponding to the model using all screened features)
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

full_model = LinearRegression()
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

# Plot the distribution of the stability scores if needed
plt.figure()
plt.hist(pred_scores_all, bins=50)
plt.title("Prediction Scores (Training Sites, Cluster {} left out)".format(test_cluster))
plt.xlabel("Prediction Score")
plt.ylabel("Frequency")
plt.savefig(f'pred_scores_cluster_{test_cluster}.png')
plt.close()

