import sys
import os
import itertools
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# Function to compute the prediction score for a given subset using XGBoost
def compute_pred_score(df_train, feature_subset, target_column):
    # Train on the entire training set using features in S.
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
    pred_score = -avg_mse  # Lower MSE gives a higher (less negative) score.
    
    return pred_score, model, scaler, y_train_min, y_train_max

if __name__ == '__main__':
    # Define test_site_index either via command-line argument or manually.
    if len(sys.argv) < 2:
        print("Usage: python loso_screened.py <test_site_index>")
        sys.exit(1)
    test_site_index = int(sys.argv[1])
    
    # Load and preprocess the data.
    df = pd.read_csv('/cluster/project/math/akmete/MSc/50sites_balanced.csv')
    df = df.dropna(axis=1, how='all')
    df = df.fillna(0)
    for col in df.select_dtypes(include=['float64']).columns:
        df[col] = df[col].astype('float32')
    # Drop columns not needed (for LOSO, we don't need 'cluster').
    df = df.drop(columns=['Unnamed: 0', 'cluster'])
    print("Loaded dataframe with columns:", df.columns)
    
    # Define initial feature columns and target.
    initial_feature_columns = [col for col in df.columns if col not in ['GPP', 'site_id']]
    target_column = "GPP"
    
    # Get all unique sites from the dataframe.
    sites = df['site_id'].unique()

    if test_site_index < 0 or test_site_index >= len(sites):
        print(f"Error: test_site_index {test_site_index} is out of range (0 to {len(sites)-1}).")
        sys.exit(1)
    test_site = sites[test_site_index]
    print(f"Processing fold (test site): {test_site}")
    
    # Split data: training = all sites except test_site, testing = test_site.
    df_train = df[df['site_id'] != test_site].copy()
    df_test  = df[df['site_id'] == test_site].copy()
    
    # === XGBOOST FEATURE SCREENING ===
    X_train_screen = df_train[initial_feature_columns]
    y_train_screen = df_train[target_column]
    
    scaler_xgb = MinMaxScaler()
    X_train_scaled_screen = scaler_xgb.fit_transform(X_train_screen)
    
    screening_model = xgb.XGBRegressor(random_state=0, n_jobs=-1)
    screening_model.fit(X_train_scaled_screen, y_train_screen)
    
    # Get feature importances and select the top 7 features.
    importances = screening_model.feature_importances_
    sorted_indices = np.argsort(importances)[::-1]
    selected_features = [initial_feature_columns[i] for i in sorted_indices[:7]]
    if not selected_features:
        raise ValueError(f"XGBoost screening removed all features for test site {test_site}.")
    print(f"Selected top 7 features for test site {test_site}: {selected_features}")
    
    # === Generate all nonempty feature subsets from the selected features ===
    all_subsets = []
    for r in range(1, len(selected_features) + 1):
        for subset in itertools.combinations(selected_features, r):
            all_subsets.append(list(subset))
    print("Number of subsets:", len(all_subsets))
    
    # === Evaluate every subset S on the training data ===
    pred_scores_all = []
    scores_info = {}
    for subset in all_subsets:
        pred_score, model, scaler, y_min, y_max = compute_pred_score(df_train, subset, target_column)
        pred_scores_all.append(pred_score)
        scores_info[str(subset)] = {
            "pred_score": pred_score,
            "model": model,
            "scaler": scaler,
            "y_min": y_min,
            "y_max": y_max
        }
    
    # === Filter subsets ===
    # Retain only those subsets whose prediction score is at or above the 95th quantile.
    alpha_pred = 0.1
    pred_threshold = np.quantile(pred_scores_all, 1 - alpha_pred)
    O_hat = [subset for subset in all_subsets if scores_info[str(subset)]["pred_score"] >= pred_threshold]
    print(f"O_hat count for test site {test_site}: {len(O_hat)}")
    
    # === Train ensemble models for each subset in O_hat using the entire training data. ===
    trained_models = {}
    for subset in O_hat:
        X_train_subset = df_train[subset]
        y_train_subset = df_train[target_column]
        scaler_model = MinMaxScaler()
        X_train_subset_scaled = scaler_model.fit_transform(X_train_subset)
        y_train_min = y_train_subset.min()
        y_train_max = y_train_subset.max()
        if y_train_max - y_train_min == 0:
            y_train_scaled = y_train_subset.values
        else:
            y_train_scaled = (y_train_subset - y_train_min) / (y_train_max - y_train_min)
        model = xgb.XGBRegressor(random_state=0, n_jobs=-1)
        model.fit(X_train_subset_scaled, y_train_scaled)
        trained_models[str(subset)] = (model, scaler_model, y_train_min, y_train_max)
    
    # === Ensemble prediction: simple average over the models in O_hat. ===
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
    
    # === Train a full model (using all selected features) for comparison. ===
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
    
    # === Evaluate on the held-out test site. ===
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
    
    # Save results for this fold.
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
        "O_hat_count": len(O_hat),
        "O_hat": O_hat
    }
    result_df = pd.DataFrame([fold_result])
    result_df.to_csv(f'results_site_{test_site}.csv', index=False)
    
    # Save a plot of prediction score distribution for this fold.
    plt.figure()
    plt.hist([scores_info[str(s)]["pred_score"] for s in all_subsets], bins=50)
    plt.title(f"Prediction Scores (Test Site {test_site})")
    plt.xlabel("Prediction Score")
    plt.ylabel("Frequency")
    plt.savefig(f'pred_scores_site_{test_site}.png')
    plt.close()

