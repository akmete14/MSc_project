import sys
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm

def main(site_arg):
    """
    site_arg: This could be either the actual site 'id' (like a string name)
              or an integer index into a list of unique sites.
    """

    # 1) Load the entire dataset (or a chunk, if you can store data per-site).
    df = pd.read_csv("/cluster/project/math/akmete/MSc/preprocessing/df_balanced_groups.csv")

    # 2) Convert float64 -> float32 for memory savings
    for col in df.select_dtypes(include=['float64']).columns:
        df[col] = df[col].astype('float32')

    # 3) Identify which column is the site ID, target, etc.
    site_col = "site_id"  # or the column that identifies each site
    target_col = "GPP"

    # For features, drop columns that are not features
    non_features = [target_col, site_col, "Unnamed: 0", "cluster"]
    features = [c for c in df.columns if c not in non_features]

    # 4) If site_arg is an *index*, we need to map it to the actual site ID
    unique_sites = df[site_col].unique()
    unique_sites = sorted(unique_sites)  # ensure reproducible ordering
    # site_arg could be integer index
    if site_arg.isdigit():
        # Convert string to int
        idx = int(site_arg)
        this_site = unique_sites[idx]
    else:
        # Otherwise, assume site_arg is already the actual site id
        this_site = site_arg

    # 5) Split train/test for this site
    test_mask = (df[site_col] == this_site)
    train_mask = ~test_mask

    X_train_raw = df.loc[train_mask, features].values
    y_train_raw = df.loc[train_mask, target_col].values
    X_test_raw = df.loc[test_mask, features].values
    y_test_raw = df.loc[test_mask, target_col].values

    if len(X_test_raw) == 0:
        print(f"Site {this_site} has no rows in dataset. Skipping.")
        return

    # 6) Scale X
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train_raw)
    X_test_scaled = scaler.transform(X_test_raw)

    X_train_scaled = np.ascontiguousarray(X_train_scaled, dtype=np.float32)
    X_test_scaled  = np.ascontiguousarray(X_test_scaled,  dtype=np.float32)

    # 7) Scale y
    y_min, y_max = y_train_raw.min(), y_train_raw.max()
    y_train_scaled = (y_train_raw - y_min) / (y_max - y_min)
    y_test_scaled  = (y_test_raw  - y_min) / (y_max - y_min)

    # 8) Train XGBoost
    model = XGBRegressor(objective='reg:squarederror', 
                         random_state=42, 
                         n_estimators=100, 
                         max_depth=5, 
                         learning_rate=0.1, 
                         n_jobs=-1)
    model.fit(X_train_scaled, y_train_scaled)

    # 9) Predict
    y_pred = model.predict(X_test_scaled)

    # 10) Metrics in scaled space
    mse = mean_squared_error(y_test_scaled, y_pred)
    r2 = r2_score(y_test_scaled, y_pred)
    relative_error = np.mean(np.abs(y_test_scaled - y_pred) / np.abs(y_test_scaled))
    mae = np.mean(np.abs(y_test_scaled - y_pred))
    rmse = np.sqrt(mse)

    # 11) Save results
    results_dict = {
        'site_left_out': this_site,
        'mse_scaled': mse,
        'r2_scaled': r2,
        'relative_error': relative_error,
        'mae_scaled': mae,
        'rmse_scaled': rmse
    }

    # Write to CSV. E.g. one row per site, or a small CSV with a unique name
    out_df = pd.DataFrame([results_dict])
    out_df.to_csv(f"results_{this_site}.csv", index=False)
    print(f"Done with site {this_site}")

if __name__ == "__main__":
    # site_arg could be e.g. "0" or "SITE123". We'll handle both
    if len(sys.argv) < 2:
        print("Usage: train_one_site.py <site_arg>")
        sys.exit(1)

    site_arg = sys.argv[1]
    main(site_arg)