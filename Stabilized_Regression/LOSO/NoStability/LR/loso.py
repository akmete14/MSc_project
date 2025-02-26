import itertools
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
import matplotlib.pyplot as plt
from joblib import Parallel, delayed

def compute_pred_score(df_train, feature_subset, target_column):
    """
    For a given feature subset S, train a regressor on all training data (i.e. all sites except the test site).
    Then, for each training site, compute the MSE (after scaling X and Y based on global training parameters).
    
    Returns:
      - pred_score: negative average MSE across training sites (so that higher is better).
      - model, scaler, y_train_min, y_train_max: these are needed later for predictions.
    """
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
    pred_score = -avg_mse  # Lower MSE gives a higher (less negative) score.
    
    return pred_score, model, scaler, y_train_min, y_train_max

if __name__ == '__main__':
    # Load and preprocess the data.
    df = pd.read_csv('/cluster/project/math/akmete/MSc/preprocessing/df_balanced_groups_onevegindex.csv')
    df = df.dropna(axis=1, how='all')
    df = df.fillna(0)
    for col in df.select_dtypes(include=['float64']).columns:
        df[col] = df[col].astype('float32')
    # Drop columns not needed. For LOSO, we don't need 'cluster'.
    df = df.drop(columns=['Unnamed: 0'])
    print("Loaded dataframe with columns:", df.columns)
    
    # Define feature columns and target.
    feature_columns = [col for col in df.columns if col not in ['GPP', 'site_id']]
    target_column = "GPP"
    
    # Generate all possible nonempty feature subsets.
    all_subsets = []
    for r in range(1, len(feature_columns) + 1):
        for subset in itertools.combinations(feature_columns, r):
            all_subsets.append(list(subset))
    print("Number of subsets:", len(all_subsets))
    
    # Loop over each fold (each site left out as test data).
    sites = sorted(df['site_id'].unique())
    overall_results = []
    
    for test_site in sites:
        print(f"\nProcessing fold (test site): {test_site}")
        df_train = df[df['site_id'] != test_site].copy()
        df_test  = df[df['site_id'] == test_site].copy()
        
        # Evaluate every subset S on the training data in parallel.
        pred_scores_all = []
        scores_info = {}  # To store information for each subset.
        
        results = Parallel(n_jobs=-1)(
            delayed(lambda s: (s, *compute_pred_score(df_train, s, target_column)))(subset)
            for subset in all_subsets
        )
        
        for subset, pred_score, model, scaler, y_min, y_max in results:
            pred_scores_all.append(pred_score)
            scores_info[str(subset)] = {
                "pred_score": pred_score,
                "model": model,
                "scaler": scaler,
                "y_min": y_min,
                "y_max": y_max
            }
        
        # Filter subsets: select only those whose prediction score is at or above the 95th quantile.
        pred_threshold = np.quantile(pred_scores_all, 0.95)
        O_hat = [subset for subset in all_subsets if scores_info[str(subset)]["pred_score"] >= pred_threshold]
        print(f"For test site {test_site}, O_hat count: {len(O_hat)}")
        
        # Train ensemble models for each subset in O_hat using the entire training data.
        trained_models = {}
        for subset in O_hat:
            X_train = df_train[subset]
            y_train = df_train[target_column]
            scaler_model = MinMaxScaler()
            X_train_scaled = scaler_model.fit_transform(X_train)
            y_train_min = y_train.min()
            y_train_max = y_train.max()
            if y_train_max - y_train_min == 0:
                y_train_scaled = y_train.values
            else:
                y_train_scaled = (y_train - y_train_min) / (y_train_max - y_train_min)
            model = LinearRegression()
            model.fit(X_train_scaled, y_train_scaled)
            trained_models[str(subset)] = (model, scaler_model, y_train_min, y_train_max)
        
        # Define ensemble prediction (simple averaging over the models in O_hat).
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
        
        # Train a full model (using all features) for comparison.
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
        
        # Evaluate on the held-out test site.
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
        
        print(f"Test Site {test_site} Ensemble MSE (scaled): {ensemble_mse_scaled}")
        print(f"Test Site {test_site} Full model MSE (scaled): {full_mse_scaled}")
        
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
        overall_results.append(fold_result)
        
        # Save plot of prediction score distribution for this fold.
        plt.figure()
        plt.hist(pred_scores_all, bins=50)
        plt.title(f"Prediction Scores (Fold: Site {test_site} left out)")
        plt.xlabel("Prediction Score")
        plt.ylabel("Frequency")
        plt.savefig(f'pred_scores_site_{test_site}.png')
        plt.close()
    
    # Save overall results for all folds.
    results_df = pd.DataFrame(overall_results)
    results_df.to_csv('results_all_folds_LOSO_prediction_only.csv', index=False)
    print("All fold results saved to results_all_folds_LOSO_prediction_only.csv")
