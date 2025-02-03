import pandas as pd
import numpy as np
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBRegressor
from tqdm import tqdm

# ---------------------------------
# Load your data
# ---------------------------------
df = pd.read_csv('/cluster/project/math/akmete/MSc/preprocessing/groupings/df_random_grouping.csv')

# For example:
features = [col for col in df.columns if col not in ["GPP", "cluster", "site_id","Unnamed: 0"]]
X = df[features].values.astype('float32')
y = df["GPP"].values.astype('float32')
groups = df["cluster"].values  # group labels

# ---------------------------------
# Leave-One-Group-Out Setup
# ---------------------------------
logo = LeaveOneGroupOut()

model = XGBRegressor(
    objective='reg:squarederror',
    random_state=42,
    n_estimators=100,
    max_depth=5,
    learning_rate=0.1,
    n_jobs=-1
)

results = []

# ---------------------------------
# Perform LOGO CV
# ---------------------------------
for train_idx, test_idx in tqdm(logo.split(X, y, groups)):
    
    # Identify training vs. testing data
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    group_train = groups[train_idx]  # groups for the training subset
    group_test  = groups[test_idx]   # groups for the test subset

    site_train = df["site_id"].values[train_idx]

    # 1) Compute sample weights for training
    sample_weight = np.zeros_like(y_train, dtype=np.float32)

    # For each group in the training fold, count how many rows it has.
    unique_train_groups = np.unique(group_train)
    for g in unique_train_groups:
        # Indices for this group
        idx_g = np.where(group_train == g)[0]
        
        # Identify the unique sites in this group
        sites_in_group = np.unique(site_train[idx_g])
        num_sites = len(sites_in_group)
        
        # Each group gets total weight = 1.0, so each site in that group gets 1 / num_sites
        site_weight = 1.0 / num_sites
        
        for site in sites_in_group:
            idx_sg = idx_g[np.where(site_train[idx_g] == site)[0]]  # Indices for this site within group g
            n_rows_site = len(idx_sg)
            
            # Weight per row in this site
            weight_per_row = site_weight / n_rows_site  # (1/num_sites) / n_rows_site
            
            sample_weight[idx_sg] = weight_per_row

    # 2) Scale y (Min-Max or Standard)
    y_min, y_max = y_train.min(), y_train.max()
    y_train_scaled = (y_train - y_min) / (y_max - y_min)
    y_test_scaled  = (y_test  - y_min) / (y_max - y_min)

    # 3) Scale X
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    # 4) Fit model with sample_weight
    model.fit(
        X_train_scaled,
        y_train_scaled,
        sample_weight=sample_weight
    )

    # 5) Predict and evaluate
    y_pred_scaled = model.predict(X_test_scaled)
    # Convert scaled predictions back to original range
    #y_pred = y_pred_scaled * (y_max - y_min) + y_min

    mse  = mean_squared_error(y_test_scaled, y_pred_scaled)
    rmse = np.sqrt(mse)
    r2   = r2_score(y_test_scaled, y_pred_scaled)
    mae  = np.mean(np.abs(y_test_scaled - y_pred_scaled))

    results.append({
        'GroupLeftOut': group_test[0],  # the single group in test fold
        'MSE':  mse,
        'RMSE': rmse,
        'R2':   r2,
        'MAE':  mae
    })

results_df = pd.DataFrame(results)
print(results_df)
results_df.to_csv("LOGOCV_WeightedTraining_weightedwithingroup_randomgrouping_group_results.csv", index=False)