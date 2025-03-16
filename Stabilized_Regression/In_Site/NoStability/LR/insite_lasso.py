# The following implementation is based on in https://arxiv.org/pdf/1911.01850. So when refering to equations we refer to equations in this paper
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

# Set random state for reproducibility (if needed)
random_state = 42

# Parse command-line argument for site index (for job array parallelism)
parser = argparse.ArgumentParser(description="Insite Stabilized Regression using Linear Regression with LASSO screening")
parser.add_argument("--site_index", type=int, required=True, help="Index of the site to process")
args = parser.parse_args()

# Load and preprocess data
df = pd.read_csv('/cluster/project/math/akmete/MSc/preprocessing/df_balanced_groups_onevegindex.csv')
df = df.dropna(axis=1, how='all')  # Drop columns where all values are NaN
df = df.fillna(0) # fill remaining NaNs with zero if there are any
df = df.drop(columns=['Unnamed: 0', 'cluster']) # drop unnecessary columns
print("Columns:", df.columns)
print("Loaded dataframe.")

# Convert float64 to float32 to save resources
for col in tqdm(df.select_dtypes(include=['float64']).columns, desc="Casting columns"):
    df[col] = df[col].astype('float32')

# Define feature and target columns
feature_columns = [col for col in df.columns if col not in ['GPP', 'site_id']]
target_column = "GPP"

# Get unique sites and select the site to process from the command-line argument
sites = sorted(df['site_id'].unique())
if args.site_index < 0 or args.site_index >= len(sites):
    raise ValueError(f"Invalid site index. Must be between 0 and {len(sites)-1}.")
site = sites[args.site_index]
print("Processing site:", site)
df_site = df[df['site_id'] == site].copy()

# For screening, split the site's data into a chronological 80/20 train-test split
split_index = int(0.8 * len(df_site))
df_train_site = df_site.iloc[:split_index]
df_test_site  = df_site.iloc[split_index:]

# Screen the features using LASSO on the training data
X_train_screen = df_train_site[feature_columns]
y_train_screen = df_train_site[target_column]

scaler_lasso = MinMaxScaler()
X_train_screen_scaled = scaler_lasso.fit_transform(X_train_screen)

lasso = LassoCV(cv=5, random_state=random_state)
lasso.fit(X_train_screen_scaled, y_train_screen)
selected_features = np.array(feature_columns)[lasso.coef_ != 0]
print("Selected features after LASSO screening:", selected_features)

# Check if any features are selected
if len(selected_features) == 0:
    raise ValueError("LASSO screening removed all features. Please check your data or adjust LASSO parameters.")



# Given screened features, define the set of all posisble subset of features
all_subsets = []
selected_features = list(selected_features)
for r in range(1, len(selected_features) + 1):
    for subset in itertools.combinations(selected_features, r):
        all_subsets.append(list(subset))
print("Number of subsets (from LASSO-selected features):", len(all_subsets))

# Define how to compute prediction and stability score
def compute_scores_insite(df_site, feature_subset, target_column):
    # Make 80/20 train-test split and define train and test for feature and target
    split_index = int(0.8 * len(df_site))
    df_train = df_site.iloc[:split_index]
    df_test  = df_site.iloc[split_index:]
    
    X_train = df_train[feature_subset]
    y_train = df_train[target_column]
    X_test  = df_test[feature_subset]
    y_test  = df_test[target_column]
    
    # Scale features
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)
    
    # Scale target
    y_train_min = y_train.min()
    y_train_max = y_train.max()

    if y_train_max - y_train_min == 0:
        y_train_scaled = y_train.values
        y_test_scaled = y_test.values
    else:
        y_train_scaled = (y_train - y_train_min) / (y_train_max - y_train_min)
        y_test_scaled = (y_test - y_train_min) / (y_train_max - y_train_min)

    
    # Define and fit model
    model = LinearRegression()
    model.fit(X_train_scaled, y_train_scaled)
    
    # Compute prediction score
    y_pred = model.predict(X_test_scaled)
    mse = mean_squared_error(y_test_scaled, y_pred)
    pred_score = -mse
    
    # compute stability score
    squared_errors = (y_test_scaled - y_pred) ** 2
    stab_score = np.var(squared_errors, ddof=1) if len(squared_errors) > 1 else 0
    
    return pred_score, stab_score, model, scaler, y_train_min, y_train_max

# Initialize to save scores
pred_scores_all = []
stab_scores_all = []

# Loop through each subset and calculate scores
for subset in tqdm(all_subsets, desc=f"Evaluating subsets for site {site}"):
    pred_score, stab_score, _, _, _, _ = compute_scores_insite(df_site, subset, target_column)
    pred_scores_all.append(pred_score)
    stab_scores_all.append(stab_score)

# Threshold for prediction score is set to .05. Consider only these subsets, having greater prediction score than the 95% quantile
alpha_pred = 0.05
c_pred = np.quantile(pred_scores_all, 1 - alpha_pred)
O_hat = [subset for subset, score in zip(all_subsets, pred_scores_all) if score >= c_pred]
print("For site", site, "O_hat count:", len(O_hat))

# Train the ensemble model as defined in Equaiton (1.1)
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
    
    model = LinearRegression()
    model.fit(X_train_scaled, y_train_scaled)
    trained_models[str(subset)] = (model, scaler, y_train_min, y_train_max)

# Each regressor in the ensemble model gets the same weight
weight = 1.0 / len(O_hat) if len(O_hat) > 0 else 0

# Define function which predicts values using ensemble model
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

# For comparison, also train the full (all variables after screening) model
X_train_full = df_train_site[selected_features]
y_train_full = df_train_site[target_column]
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

# Predict on the 20% test data and get metrics
X_test_full = df_test_site[selected_features]
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

ensemble_preds_scaled = ensemble_predict(df_test_site, return_scaled=True)
ensemble_mse_scaled = mean_squared_error(y_test_scaled_full, ensemble_preds_scaled)
ensemble_rmse_scaled = np.sqrt(ensemble_mse_scaled)
ensemble_r2 = r2_score(y_test_scaled_full, ensemble_preds_scaled)
ensemble_relative_error = np.mean(np.abs(y_test_scaled_full - ensemble_preds_scaled) / np.abs(y_test_scaled_full))
ensemble_mae = np.mean(np.abs(y_test_scaled_full - ensemble_preds_scaled))

print("Site {} Ensemble MSE (scaled): {}".format(site, ensemble_mse_scaled))
print("Site {} Full model MSE (scaled): {}".format(site, full_mse_scaled))

# Save results from ensemble model and full model
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
    "O_hat_count": len(O_hat)
}

# Save results to a csv
results_df = pd.DataFrame([results])
results_df.to_csv(f'results_site_{site}.csv', index=False)
print("Results saved to results_site_{}.csv".format(site))

# For comparing prediction scores, plot for every 10th site the distribution of the prediction scores
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
