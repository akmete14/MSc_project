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

# Define function computing pred and stab score
def compute_scores(df_train, feature_subset, target_column):

    # Train on all training data using features in S.
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

        # Transform using the scaler fitted on the entire training set.
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

# Main function
if __name__ == '__main__':
    # Parse the command-line argument for test cluster.
    parser = argparse.ArgumentParser(description="LOGO Stabilized Regression using Linear Regression with LASSO screening")
    parser.add_argument("--test_cluster", type=int, required=True, help="Cluster to leave out for testing")
    args = parser.parse_args()
    
    # Load and preprocess  data.
    df = pd.read_csv('/cluster/project/math/akmete/MSc/preprocessing/df_balanced_groups_onevegindex.csv')
    print("Unique clusters:", df['cluster'].unique())
    df = df.dropna(axis=1, how='all') # drop rows where only NaNs
    df = df.fillna(0) # fill NaNs with 0 if there are any
    df = df.drop(columns=['Unnamed: 0']) # Drop unnecessary files
    print("Loaded dataframe with columns:", df.columns)

    # Convert float64 to float32 to save resources
    for col in df.select_dtypes(include=['float64']).columns:
        df[col] = df[col].astype('float32')
    
    # Define feature and target columns
    initial_feature_columns = [col for col in df.columns if col not in ['GPP', 'site_id', 'cluster']]
    target_column = "GPP"
    
    # Use the test_cluster provided in the command-line argument.
    test_cluster = int(args.test_cluster)
    print("\nProcessing fold (test cluster):", test_cluster)
    df_train = df[df['cluster'] != test_cluster].copy()
    df_test  = df[df['cluster'] == test_cluster].copy()
    
    # SCREENING
    X_train = df_train[initial_feature_columns]
    y_train = df_train[target_column]
    
    scaler_lasso = MinMaxScaler()
    X_train_scaled = scaler_lasso.fit_transform(X_train)
    
    lasso = LassoCV(cv=5, random_state=0)
    lasso.fit(X_train_scaled, y_train)
    
    selected_features = np.array(initial_feature_columns)[lasso.coef_ != 0]
    selected_features = list(selected_features)
    if not selected_features:
        raise ValueError(f"LASSO screening removed all features for test cluster {test_cluster}.")
    print("For test cluster", test_cluster, "selected features:", selected_features)
    

    # Given screened variables, generate all possible subsets of features
    all_subsets = []
    for r in range(1, len(selected_features) + 1):
        for subset in itertools.combinations(selected_features, r):
            all_subsets.append(list(subset))
    print("Number of subsets:", len(all_subsets))
    
    # For each of these sets, we calculate the scores and save them
    pred_scores_all = []
    stability_scores_all = []
    scores_info = {}
    
    for subset in tqdm(all_subsets, desc=f"Evaluating subsets for cluster {test_cluster}"):
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
    
    # Keep only those subsets, that satisfy stability threshold
    alpha_stab = np.quantile(stability_scores_all, 0.1)
    G_hat = [subset for subset, stab in zip(all_subsets, stability_scores_all) if stab <= alpha_stab]
    print("For test cluster", test_cluster, "G_hat count:", len(G_hat))
    
    # After filtering wrt to stability, get from these subsets only those satisfying the prediction threshold
    alpha_pred = 0.05
    g_hat_pred_scores = [score for subset, score in zip(all_subsets, pred_scores_all) if subset in G_hat]
    c_pred = np.quantile(g_hat_pred_scores,1-alpha_pred)
    O_hat = [subset for subset, score in zip(all_subsets, pred_scores_all) if (subset in G_hat) and (score>=c_pred)]
    print("For test cluster", test_cluster, "O_hat count:", len(O_hat))
    
    # Train the ensemble model
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
        model = LinearRegression()
        model.fit(X_train_scaled, y_train_scaled)
        trained_models[str(subset)] = (model, scaler_subset, y_train_min, y_train_max)
    
    # Every regressor in the ensemble model gets the same weight
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
    
    # For compare the performance of SR, we train the full model on all screened features
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
    
    # Evaluate ensemble model and full model and get all metrics
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
    
    print("Test Cluster {} Ensemble MSE (scaled): {}".format(test_cluster, ensemble_mse_scaled))
    print("Test Cluster {} Full model MSE (scaled): {}".format(test_cluster, full_mse_scaled))
    
    # Save results for this fold
    fold_result = {
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
    # Save results of this fold
    results_df = pd.DataFrame([fold_result])
    results_df.to_csv(f'results_cluster_{test_cluster}.csv', index=False)
    print("Results saved to results_cluster_{}.csv".format(test_cluster))
    
    # Save plots
    plt.figure()
    plt.hist(pred_scores_all, bins=50)
    plt.title("Prediction Scores (Training Sites, Cluster {} left out)".format(test_cluster))
    plt.xlabel("Prediction Score")
    plt.ylabel("Frequency")
    plt.savefig(f'pred_scores_cluster_{test_cluster}.png')
    plt.close()
    
    plt.figure()
    plt.hist(stability_scores_all, bins=50)
    plt.title("Stability Scores (Training Sites, Cluster {} left out)".format(test_cluster))
    plt.xlabel("Stability Score")
    plt.ylabel("Frequency")
    plt.savefig(f'stability_scores_cluster_{test_cluster}.png')
    plt.close()

