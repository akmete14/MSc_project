import pandas as pd
import numpy as np
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from tqdm import tqdm

# Load data
df_train = pd.read_csv('/cluster/project/math/akmete/MSc/preprocessing/train_data.csv')
df_test = pd.read_csv('/cluster/project/math/akmete/MSc/preprocessing/test_data.csv')
print(len(df_train), len(df_test), df_train.head(), df_test.head(), df_train.columns)

# Reduce memory to float32
for col in tqdm(df_train.select_dtypes(include=['float64']).columns):
    df_train[col] = df_train[col].astype('float32')

for col in tqdm(df_test.select_dtypes(include=['float64']).columns):
    df_test[col] = df_test[col].astype('float32')

print("reduced to float32")

# Define features and target
feature_cols = [
    col for col in df_train.columns
    if col not in ["GPP", "site_id", "cluster","Unnamed: 0"]
]
target_col = "GPP"

X_train = df_train[feature_cols]
y_train = df_train[target_col]

print("now start scaling")
# Scale X_train and y_train
X_scaler = MinMaxScaler()
X_train_scaled = X_scaler.fit_transform(X_train)
y_train_standard = (y_train - y_train.mean()) / (y_train.max() - y_train.min())
print("scaled")

# Ensure contiguous array in float32 format
X_train_scaled = np.ascontiguousarray(X_train_scaled, dtype=np.float32)

# Initialize and train model
model = XGBRegressor(
    objective='reg:squarederror', 
    random_state=42, 
    n_estimators=100, 
    max_depth=5, 
    learning_rate=0.1, 
    n_jobs=-1
)
model.fit(X_train_scaled, y_train_standard)
print("model fitted")

# Initialize list to collect results
results = []

# Group df_test by 'cluster' and evaluate each cluster
for cluster_id, cluster_data in tqdm(df_test.groupby('cluster')):
    # Extract features and target for the current cluster
    X_cluster = cluster_data[feature_cols]
    y_cluster = cluster_data[target_col]
    
    # Scale features using the training scaler
    X_cluster_scaled = X_scaler.transform(X_cluster)
    y_cluster_standard = (y_cluster - y_train.mean()) / (y_train.max() - y_train.min())
    
    # Ensure contiguous array in float32 format
    X_cluster_scaled = np.ascontiguousarray(X_cluster_scaled, dtype=np.float32)

    # Predict using the model
    y_pred = model.predict(X_cluster_scaled)
    
    # Compute evaluation metrics
    mse = mean_squared_error(y_cluster_standard, y_pred)
    r2 = r2_score(y_cluster_standard, y_pred)
    
    # Save metrics in results
    results.append({
        'cluster': cluster_id,
        'mse': mse,
        'r2': r2,
        'num_samples': len(cluster_data)
    })

    # -------------------
    # Plot 1: Jointplot of True vs. Predicted for this cluster
    # -------------------
    g = sns.jointplot(x=y_cluster_standard, y=y_pred, kind='scatter', s=1)
    # Add y=x line for reference
    g.ax_joint.plot([-0.4, 0.4], [-0.4, 0.4], color='red')
    plt.suptitle(f"XGBoost - Cluster: {cluster_id}", y=1.02)  # Adjust y for title placement
    g.set_axis_labels("True GPP", "Predicted GPP")
    plt.tight_layout()

    # Save the jointplot figure with a unique filename
    plt.savefig(f'TRUEvsPREDICTED_distr_XGB_cluster_{cluster_id}.png', dpi=300, bbox_inches='tight')
    plt.close()

    # -------------------
    # Plot 2: Histogram of True vs. Predicted for this cluster
    # -------------------
    plt.figure(figsize=(6, 4))
    bins = np.linspace(-0.3, 0.4, 100)
    plt.hist(y_cluster_standard, bins=bins, alpha=0.5, label='True GPP')
    plt.hist(y_pred, bins=bins, alpha=0.5, label='Predicted GPP')
    plt.title(f"XGBoost - Cluster: {cluster_id}")
    plt.legend()
    plt.tight_layout()

    # Save the histogram figure with a unique filename
    plt.savefig(f'TRUEvsPREDICTED_hist_XGB_cluster_{cluster_id}.png', dpi=300, bbox_inches='tight')
    plt.close()

# Convert results list to DataFrame and save CSV
results_df = pd.DataFrame(results)
results_df.to_csv('Metrics_MinimalDomainShift_XGB_withplots.csv', index=False)
print("Evaluation Metrics by Cluster:")
print(results_df)
