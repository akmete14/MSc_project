import pandas as pd
import numpy as np
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
import xarray as xr
from xgboost import XGBRegressor
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

# Load data and reduce float64 to float32
df = pd.read_csv('/cluster/project/math/akmete/MSc/preprocessing/df_balanced_groups.csv')
for col in tqdm(df.select_dtypes(include=['float64']).columns):
    df[col] = df[col].astype('float32')
print(df.columns)

# Assume df is your dataframe with columns: 'target', 'grouping', and features
features = [col for col in df.columns if col not in ["GPP", "cluster", "site_id","Unnamed: 0"]]
X = df[features].values  # will already be float32 if the df columns are float32
y = df["GPP"].values     # consider downcasting if it’s float64
groups = df["cluster"].values

logo = LeaveOneGroupOut()
results = []

# Initialize model
model = XGBRegressor(objective='reg:squarederror', random_state=42, n_estimators=100, max_depth=5, learning_rate=0.1, n_jobs=-1)

# Perform LOGO CV
for i, (train_idx, test_idx) in enumerate(tqdm(logo.split(X, y, groups))):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # Scale y (min-max scaling)
    y_train_standard = (y_train - y_train.mean()) / (y_train.max() - y_train.min())
    y_test_standard = (y_test - y_train.mean()) / (y_train.max() - y_train.min())
    
    # Scale X (min-max scaling)
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Convert to float32
    X_train_scaled = np.ascontiguousarray(X_train_scaled, dtype=np.float32)
    X_test_scaled  = np.ascontiguousarray(X_test_scaled,  dtype=np.float32)
    
    # Fit model
    model.fit(X_train_scaled, y_train_standard)
    
    # Predict and evaluate
    y_pred = model.predict(X_test_scaled)
    mse = mean_squared_error(y_test_standard, y_pred)
    
    # Store results
    group_out = groups[test_idx][0]  # This identifies the group that's held out
    results.append({'group_left_out': group_out, 'mse': mse})

    # -------------------
    # Plot 1: Jointplot of True vs. Predicted
    # -------------------
    # We use a figure-level function (sns.jointplot) so let's close the figure afterwards
    g = sns.jointplot(x=y_test_standard, y=y_pred, kind='scatter', s=1)
    g.ax_joint.plot([-0.4, 0.4], [-0.4, 0.4], color='red')  # y=x line on the joint plot
    plt.title(f"XGBoost - Group left out: {group_out}")
    g.set_axis_labels("True GPP", "Predicted GPP")
    # Save with a unique filename
    plt.savefig(f'TRUEvsPREDICTED_distr_XGB_{group_out}.png', dpi=300, bbox_inches='tight')
    plt.close()

    # -------------------
    # Plot 2: Histogram comparison of True vs. Predicted
    # -------------------
    plt.figure(figsize=(6, 4))
    bins = np.linspace(-0.3, 0.4, 100)
    plt.title(f"XGBoost - Group left out: {group_out}")
    plt.hist(y_test_standard, bins=bins, alpha=0.5, label='True GPP')
    plt.hist(y_pred, bins=bins, alpha=0.5, label='Predicted GPP')
    plt.legend()
    plt.savefig(f'TRUEvsPREDICTED_hist_XGB_{group_out}.png', dpi=300, bbox_inches='tight')
    plt.close()

# Convert results to a DataFrame for better visualization
results_df = pd.DataFrame(results)
print(results_df)

# Optionally, you can save the CSV of results:
results_df.to_csv('LOGOCVXGB_100hrs_withplots.csv', index=False)