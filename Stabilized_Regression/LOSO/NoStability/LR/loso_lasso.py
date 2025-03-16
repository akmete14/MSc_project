# Import libraries
import os
import itertools
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, LassoCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# Define function that computes prediction score for a given subset S
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
    
    model = LinearRegression()
    model.fit(X_train_scaled, y_train_scaled)
    
    # Compute per-site MSEs.
    training_sites = df_train['site_id'].unique()
    mse_list = []
    for site in training_sites:
        df_site = df_train[df_train['site_id'] == site]
        X_site = df_site[feature_subset]
        y_site = df_site[target_column]
        
        # Scale using the global scaler from the full training data.
        X_site_scaled = scaler.transform(X_site)
        if y_train_max - y_train_min == 0:
            y_site_scaled = y_site.values
        else:
            y_site_scaled = (y_site - y_train_min) / (y_train_max - y_train_min)
        
        y_pred = model.predict(X_site_scaled)
        mse = mean_squared_error(y_site_scaled, y_pred)
        mse_list.append(mse)
    
    avg_mse = np.mean(mse_list)
    pred_score = -avg_mse  # Calculate prediction score
    
    return pred_score, model, scaler, y_train_min, y_train_max

# Main function
if __name__ == '__main__':
    # Load and preprocess the data.
    df = pd.read_csv('/cluster/project/math/akmete/MSc/preprocessing/df_balanced_groups_onevegindex.csv')
    df = df.dropna(axis=1, how='all') # remove columns when containing only NaNs
    df = df.fillna(0) # fill NaNs with 0 if there are any
    df = df.drop(columns=['Unnamed: 0', 'cluster']) # drop unnecessary columns
    print("Loaded dataframe with columns:", df.columns)

    # Convert float64 to float32 to save memory
    for col in df.select_dtypes(include=['float64']).columns:
        df[col] = df[col].astype('float32')
    
    # Define initial feature and target columns (initial because we remove some after screening)
    initial_feature_columns = [col for col in df.columns if col not in ['GPP', 'site_id']]
    target_column = "GPP"
    
    # Get unique sites.
    sites = sorted(df['site_id'].unique())
    
    # Use SLURM_ARRAY_TASK_ID to choose which test site to process.
    array_task = int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))
    try:
        test_site = sites[array_task]
    except IndexError:
        raise ValueError("SLURM_ARRAY_TASK_ID is out of range. Check the number of unique sites.")
    print(f"Processing fold for test site: {test_site}")
    
    # Split data into train and test for this fold
    df_train = df[df['site_id'] != test_site].copy()
    df_test  = df[df['site_id'] == test_site].copy()
    
    # SCREENING using LASSSO
    X_train = df_train[initial_feature_columns]
    y_train = df_train[target_column]
    
    scaler_lasso = MinMaxScaler()
    X_train_scaled = scaler_lasso.fit_transform(X_train)
    
    lasso = LassoCV(cv=5, random_state=0)
    lasso.fit(X_train_scaled, y_train)
    
    selected_features = np.array(initial_feature_columns)[lasso.coef_ != 0]
    selected_features = list(selected_features)
    if not selected_features:
        raise ValueError(f"LASSO screening removed all features for test site {test_site}.")
    print(f"Selected features for test site {test_site}: {selected_features}")
    
    # Given screened feature set, generate all possible sets of features
    all_subsets = []
    for r in range(1, len(selected_features) + 1):
        for subset in itertools.combinations(selected_features, r):
            all_subsets.append(list(subset))
    print("Number of subsets:", len(all_subsets))
    
    # For each set, get prediction score
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
    
    # Filter with respect to prediction threshold
    alpha_pred = 0.05
    pred_threshold = np.quantile(pred_scores_all, 1-alpha_pred)
    O_hat = [subset for subset in all_subsets if scores_info[str(subset)]["pred_score"] >= pred_threshold]
    print(f"O_hat count for test site {test_site}: {len(O_hat)}")
    
    # Given O_hat, train ensemble model
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
        model = LinearRegression()
        model.fit(X_train_scaled, y_train_scaled)
        trained_models[str(subset)] = (model, scaler_model, y_train_min, y_train_max)
    
    # Get weight for each regressor in ensemble model
    weight = 1.0 / len(O_hat) if len(O_hat) > 0 else 0

    # Define prediction function for the ensemble model
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
    
    # Train a model using all screened features so that we can compare to ensemble model
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
    full_model = LinearRegression()
    full_model.fit(X_train_full_scaled, y_train_full_scaled)
    
    # Evalaute on test site
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
    
    # Save results for ensemble and full model (and then to CSV)
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
        "O_hat_count": len(O_hat)
    }
    result_df = pd.DataFrame([fold_result])
    result_df.to_csv(f'results_site_{test_site}.csv', index=False)
    
    # To compare prediction score distributions, you can plot them
    import matplotlib.pyplot as plt
    plt.figure()
    plt.hist([scores_info[str(s)]["pred_score"] for s in all_subsets], bins=50)
    plt.title(f"Prediction Scores (Test Site {test_site})")
    plt.xlabel("Prediction Score")
    plt.ylabel("Frequency")
    plt.savefig(f'pred_scores_site_{test_site}.png')
    plt.close()