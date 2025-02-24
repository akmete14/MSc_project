import argparse
import itertools
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
import matplotlib.pyplot as plt

# Set random state for reproducibility
random_state = 42

# Parse command-line argument for fold index
parser = argparse.ArgumentParser(description="LOSO Stabilized Regression Fold")
parser.add_argument("--fold", type=int, required=True, help="Fold index (0, 1, or 2)")
args = parser.parse_args()

# Read data and preprocess
df = pd.read_csv('/cluster/project/math/akmete/MSc/Stabilized_Regression/LOSO/5_sites_onevegindex/dataframe_5_sites_test_onevegindex.csv')
df = df.dropna(axis=1, how='all')  # Drop columns where all values are NaN
df = df.fillna(0)
for col in tqdm(df.select_dtypes(include=['float64']).columns):
    df[col] = df[col].astype('float32')
df = df.drop(columns=['Unnamed: 0', 'cluster', 'NDWI_band7', 'NIRv', 'EVI', 'hour'])
print("Columns:", df.columns)
print("Loaded dataframe.")

# Define features and target
feature_columns = [col for col in df.columns if col not in ['GPP', 'site_id']]
target_column = "GPP"

# Create all feature subsets
all_subsets = []
for r in range(1, len(feature_columns) + 1):
    for subset in itertools.combinations(feature_columns, r):
        all_subsets.append(list(subset))
print("Number of subsets:", len(all_subsets))

# Get unique sites and select the test site based on fold index
sites = sorted(df['site_id'].unique())
print("Unique sites:", sites)
if args.fold < 0 or args.fold >= len(sites):
    raise ValueError("Invalid fold index. Must be between 0 and {}.".format(len(sites)-1))
test_site = sites[args.fold]
train_sites = [s for s in sites if s != test_site]
print("Running fold:", args.fold, "with test site:", test_site, "and training sites:", train_sites)

# Function to compute scores for a given feature subset and fold
def compute_scores(df, feature_subset, target_column, train_sites, test_site):
    df_train = df[df['site_id'].isin(train_sites)]
    df_test = df[df['site_id'] == test_site]
    
    X_train = df_train[feature_subset]
    y_train = df_train[target_column]
    X_test = df_test[feature_subset]
    y_test = df_test[target_column]
    
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
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
    
    mse_list = []
    for site in train_sites:
        df_site = df_train[df_train['site_id'] == site]
        X_site = df_site[feature_subset]
        y_site = df_site[target_column]
        X_site_scaled = scaler.transform(X_site)
        if y_train_max - y_train_min == 0:
            y_site_scaled = y_site.values
        else:
            y_site_scaled = (y_site - y_train_min) / (y_train_max - y_train_min)
        y_pred_site = model.predict(X_site_scaled)
        mse_site = mean_squared_error(y_site_scaled, y_pred_site)
        mse_list.append(mse_site)
    
    pred_score = -np.mean(mse_list)
    stab_score = np.var(mse_list, ddof=1) if len(mse_list) > 1 else 0
    return pred_score, stab_score, model, scaler, y_train_min, y_train_max

# Set threshold parameters
#alpha_pred = 0.05
#quantile_level = 0.75  # select subsets above the 75th percentile of stability scores
#alpha_stab = np.quantile(stab_scores_all, quantile_level)

# Compute scores for each subset on the current fold
pred_scores_all = []
stab_scores_all = []
G_hat = []

for subset in tqdm(all_subsets, desc="Evaluating subsets"):
    pred_score, stab_score, _, _, _, _ = compute_scores(df, subset, target_column, train_sites, test_site)
    pred_scores_all.append(pred_score)
    stab_scores_all.append(stab_score)
    #if stab_score >= alpha_stab:
        #G_hat.append(subset)

# Now set threshold parameters using data-driven approaches:
alpha_pred = 0.05
quantile_level = 0.75  # for example, select subsets above the 75th percentile of stability scores
alpha_stab = np.quantile(stab_scores_all, quantile_level)
print("Stability threshold (alpha_stab):", alpha_stab)

# Given this cutoff, get G_hat:
G_hat = [subset for subset, score in zip(all_subsets, stab_scores_all) if score >= alpha_stab]

plt.figure()
plt.hist(stab_scores_all, bins=50)
plt.title("Stability Scores (Test site: {})".format(test_site))
plt.savefig('stab_scores_{}.png'.format(test_site))

plt.figure()
plt.hist(pred_scores_all, bins=50)
plt.title("Prediction Scores (Test site: {})".format(test_site))
plt.savefig('pred_scores_{}.png'.format(test_site))

print("Min/Max stab_scores:", min(stab_scores_all), max(stab_scores_all))
print("Min/Max pred_scores:", min(pred_scores_all), max(pred_scores_all))

c_pred = np.quantile(pred_scores_all, 1 - alpha_pred)
O_hat = [subset for subset, score in zip(all_subsets, pred_scores_all) if score >= c_pred]

print("Selected subsets meeting stability (G_hat):", G_hat)
print("Selected subsets meeting prediction (O_hat):", O_hat)

# Train ensemble models for subsets in O_hat on the training sites
trained_models = {}
for subset in O_hat:
    df_train_fold = df[df['site_id'].isin(train_sites)]
    X_train = df_train_fold[subset]
    y_train = df_train_fold[target_column]
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
    """
    If return_scaled is True, returns the raw scaled predictions from each model;
    otherwise, returns predictions transformed back to the original scale.
    """
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

# --- Evaluate on the test site ---
df_test_fold = df[df['site_id'] == test_site]

# For the full model, we train using all features from the training sites:
df_train_fold = df[df['site_id'].isin(train_sites)]
X_train_full = df_train_fold[feature_columns]
y_train_full = df_train_fold[target_column]
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

# Evaluate on test data for full model
df_test_fold = df[df['site_id'] == test_site]
X_test_full = df_test_fold[feature_columns]
X_test_full_scaled = scaler_full.transform(X_test_full)
full_preds_scaled = full_model.predict(X_test_full_scaled)
# Compute full model MSE in scaled space (ground truth scaled using training parameters)
if y_train_max_full - y_train_min_full == 0:
    y_test_scaled_full = df_test_fold[target_column].values
else:
    y_test_scaled_full = (df_test_fold[target_column].values - y_train_min_full) / (y_train_max_full - y_train_min_full)
full_mse_scaled = mean_squared_error(y_test_scaled_full, full_preds_scaled)

# Also compute the ensemble predictions:
# a) Original scale (for reference)
ensemble_preds_original = ensemble_predict(df_test_fold, return_scaled=False)
ensemble_mse_original = mean_squared_error(df_test_fold[target_column].values, ensemble_preds_original)
# b) Scaled space: we compute the raw ensemble predictions and compare them with y_test scaled using full model's parameters
ensemble_preds_scaled = ensemble_predict(df_test_fold, return_scaled=True)
ensemble_mse_scaled = mean_squared_error(y_test_scaled_full, ensemble_preds_scaled)

print("Ensemble MSE on test site {} (original scale): {}".format(test_site, ensemble_mse_original))
print("Ensemble MSE on test site {} (scaled): {}".format(test_site, ensemble_mse_scaled))
print("Full model MSE on test site {} (scaled): {}".format(test_site, full_mse_scaled))
print(len(G_hat))
print(len(O_hat))