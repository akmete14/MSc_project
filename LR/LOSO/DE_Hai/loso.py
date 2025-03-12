import os
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# -----------------------------
# 1) Load and Preprocess Data
# -----------------------------
df = pd.read_csv('/cluster/project/math/akmete/MSc/preprocessing/df_balanced_groups_onevegindex.csv')
df = df.dropna(axis=1, how='all')  # Drop columns where all values are NaN
df = df.fillna(0)
df = df.drop(columns=['Unnamed: 0', 'cluster'])

# Cast float64 columns to float32 for efficiency
for col in tqdm(df.select_dtypes(include=['float64']).columns, desc="Casting columns"):
    df[col] = df[col].astype('float32')

# Define features and target
feature_columns = [col for col in df.columns if col not in ['GPP', 'site_id']]
target_column = "GPP"

# -----------------------------
# 2) Define the 'DE-Hai' Fold
# -----------------------------
test_site = 'DE-Hai'
df_train = df[df['site_id'] != test_site].copy()
df_test  = df[df['site_id'] == test_site].copy()

# -----------------------------
# 3) Scaling
# -----------------------------
# Scale features based on training data only
scaler_X = MinMaxScaler()
X_train = df_train[feature_columns]
X_test  = df_test[feature_columns]
X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled  = scaler_X.transform(X_test)

# Scale target variable using training data values
y_train = df_train[target_column]
y_test  = df_test[target_column]

y_train_min = y_train.min()
y_train_max = y_train.max()
if y_train_max - y_train_min == 0:
    y_train_scaled = y_train.values
    y_test_scaled  = y_test.values
else:
    y_train_scaled = (y_train - y_train_min) / (y_train_max - y_train_min)
    y_test_scaled  = (y_test - y_train_min) / (y_train_max - y_train_min)

# Convert arrays to float32
X_train_scaled = np.asarray(X_train_scaled, dtype=np.float32)
X_test_scaled  = np.asarray(X_test_scaled, dtype=np.float32)
y_train_scaled = np.asarray(y_train_scaled, dtype=np.float32)

# -----------------------------
# 4) Train the Linear Regression Model
# -----------------------------
model = LinearRegression()
model.fit(X_train_scaled, y_train_scaled)

# -----------------------------
# 5) Prediction and Metrics
# -----------------------------
y_pred = model.predict(X_test_scaled)

mse = mean_squared_error(y_test_scaled, y_pred)
r2  = r2_score(y_test_scaled, y_pred)
rmse = np.sqrt(mse)
relative_error = np.mean(np.abs(y_test_scaled - y_pred) / np.abs(y_test_scaled))
mae = np.mean(np.abs(y_test_scaled - y_pred))

print(f"Test Site {test_site}: MSE={mse:.6f}, R2={r2:.6f}, RMSE={rmse:.6f}, RelError={relative_error:.6f}, MAE={mae:.6f}")

# -----------------------------
# 6) Save Actual and Predicted Values
# -----------------------------
results_df = pd.DataFrame({
    'actual': y_test_scaled,
    'predicted': y_pred
})
results_df.to_csv("results_DE-Hai_linear.csv", index=False)
print("Results saved to results_DE-Hai_linear.csv")

# -----------------------------
# 7) Joint Scatter Plot with Marginal Histograms
# -----------------------------
y_test_standard = results_df["actual"]
y_pred_values   = results_df["predicted"]

# Create the joint scatter plot with histograms on the margins
g = sns.jointplot(x=y_test_standard, y=y_pred_values, kind="scatter", s=1)

# Add the y=x reference line
plt.plot([0.0, 1.0], [0.0, 1.0], color="red")

# Set axis labels
plt.xlabel("True GPP")
plt.ylabel("Predicted GPP")

# Save the plot to a file and display it
plt.savefig('de_hai_lr.png')