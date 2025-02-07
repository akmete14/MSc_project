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

    # 1) Load the entire dataset
    df = pd.read_csv("/cluster/project/math/akmete/MSc/preprocessing/df_balanced_groups.csv")

    # 2) Convert float64 -> float32 for memory savings
    for col in df.select_dtypes(include=['float64']).columns:
        df[col] = df[col].astype('float32')

    # 3) Identify columns: site id, target, and features.
    site_col = "site_id"  # column that identifies each site
    target_col = "GPP"
    non_features = [target_col, site_col, "Unnamed: 0", "cluster"]
    features = [c for c in df.columns if c not in non_features]

    # 4) Map site_arg to the actual site id
    unique_sites = sorted(df[site_col].unique())
    if site_arg.isdigit():
        idx = int(site_arg)
        this_site = unique_sites[idx]
    else:
        this_site = site_arg

    # 5) Create train/test masks: test is the left-out site.
    test_mask = (df[site_col] == this_site)
    train_mask = ~test_mask

    # 6) For independent scaling, process the training data by site.
    train_df = df.loc[train_mask].copy()
    # Create an empty list to store scaled training data
    scaled_train_list = []

    # Group training data by site and scale each group independently
    for site, group in train_df.groupby(site_col):
        X_site = group[features].values
        y_site = group[target_col].values

        # Scale features using a separate scaler for this site
        scaler_X = MinMaxScaler()
        X_site_scaled = scaler_X.fit_transform(X_site)

        # Scale target independently (using per-site min/max)
        y_min, y_max = y_site.min(), y_site.max()
        # Avoid division by zero in case y_max equals y_min
        if y_max == y_min:
            y_site_scaled = np.zeros_like(y_site)
        else:
            y_site_scaled = (y_site - y_min) / (y_max - y_min)

        # Create a DataFrame with the scaled values and preserve the original indices
        scaled_site_df = pd.DataFrame(X_site_scaled, columns=features, index=group.index)
        scaled_site_df[target_col] = y_site_scaled
        scaled_train_list.append(scaled_site_df)

    # Concatenate all the independently scaled training data
    scaled_train_df = pd.concat(scaled_train_list)
    # Reconstruct training arrays preserving the original ordering
    X_train_scaled = scaled_train_df.loc[train_df.index, features].values
    y_train_scaled = scaled_train_df.loc[train_df.index, target_col].values

    # 7) Scale test data independently (test data comes from one site)
    test_df = df.loc[test_mask].copy()
    scaler_X_test = MinMaxScaler()
    X_test_scaled = scaler_X_test.fit_transform(test_df[features].values)
    y_test_raw = test_df[target_col].values
    y_min_test, y_max_test = y_test_raw.min(), y_test_raw.max()
    if y_max_test == y_min_test:
        y_test_scaled = np.zeros_like(y_test_raw)
    else:
        y_test_scaled = (y_test_raw - y_min_test) / (y_max_test - y_min_test)

    # Ensure data is contiguous and in float32
    X_train_scaled = np.ascontiguousarray(X_train_scaled, dtype=np.float32)
    X_test_scaled  = np.ascontiguousarray(X_test_scaled,  dtype=np.float32)
    y_train_scaled = np.ascontiguousarray(y_train_scaled, dtype=np.float32)
    y_test_scaled  = np.ascontiguousarray(y_test_scaled,  dtype=np.float32)

    # 8) Train XGBoost
    model = XGBRegressor(
        objective='reg:squarederror', 
        random_state=42, 
        n_estimators=100, 
        max_depth=5, 
        learning_rate=0.1, 
        n_jobs=-1
    )
    model.fit(X_train_scaled, y_train_scaled)

    # 9) Predict
    y_pred = model.predict(X_test_scaled)

    # 10) Compute metrics in scaled space
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
    out_df = pd.DataFrame([results_dict])
    out_df.to_csv(f"results_{this_site}.csv", index=False)
    print(f"Done with site {this_site}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: train_one_site.py <site_arg>")
        sys.exit(1)
    site_arg = sys.argv[1]
    main(site_arg)
