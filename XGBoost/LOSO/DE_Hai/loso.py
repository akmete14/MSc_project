#Import libraries
import os
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBRegressor
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# Load and process data
df = pd.read_csv('/cluster/project/math/akmete/MSc/preprocessing/df_balanced_groups_onevegindex.csv')
df = df.dropna(axis=1, how='all')  # Drop columns where all values are NaN
df = df.fillna(0) # fill NaN with 0 if there are any left
df = df.drop(columns=['Unnamed: 0', 'cluster']) # drop unnecessary columns

# Convert float64 to float32 to save memory
for col in tqdm(df.select_dtypes(include=['float64']).columns, desc="Casting columns"):
    df[col] = df[col].astype('float32')

# Define features and target
feature_columns = [col for col in df.columns if col not in ['GPP', 'site_id']]
target_column = "GPP"

# select the site we want to test predict on
test_site = 'DE-Hai'
df_train = df[df['site_id'] != test_site].copy()
df_test  = df[df['site_id'] == test_site].copy()

# Scale features X
scaler_X = MinMaxScaler()
X_train = df_train[feature_columns]
X_test  = df_test[feature_columns]
X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled  = scaler_X.transform(X_test)

# Define and scale target y
y_train = df_train[target_column]
y_test  = df_test[target_column]

y_train_min = y_train.min()
y_train_max = y_train.max()

if y_train_max - y_train_min == 0:
    y_train_scaled = y_train.values
    y_test_scaled = y_test.values
else:
    y_train_scaled = (y_train - y_train_min) / (y_train_max - y_train_min)
    y_test_scaled  = (y_test - y_train_min) / (y_train_max - y_train_min)

# Convert arrays to float32
X_train_scaled = np.asarray(X_train_scaled, dtype=np.float32)
X_test_scaled  = np.asarray(X_test_scaled, dtype=np.float32)
y_train_scaled = np.asarray(y_train_scaled, dtype=np.float32)

# Define and train model
model = XGBRegressor(objective='reg:squarederror',
                     n_estimators=100,
                     max_depth=5,
                     learning_rate=0.1,
                     random_state=42)
model.fit(X_train_scaled, y_train_scaled)

# get prediction and calculate metrics
y_pred = model.predict(X_test_scaled)

mse = mean_squared_error(y_test_scaled, y_pred)
r2  = r2_score(y_test_scaled, y_pred)
rmse = np.sqrt(mse)
relative_error = np.mean(np.abs(y_test_scaled - y_pred) / np.abs(y_test_scaled))
mae = np.mean(np.abs(y_test_scaled - y_pred))

print(f"Test Site {test_site}: MSE={mse:.6f}, R2={r2:.6f}, RMSE={rmse:.6f}, RelError={relative_error:.6f}, MAE={mae:.6f}")

# Save results to csv
results_df = pd.DataFrame({
    'actual': y_test_scaled,
    'predicted': y_pred
})
results_df.to_csv("results_DE-Hai.csv", index=False)
print("Results saved to results_DE-Hai.csv")

# get actual and predicted values to make plots later
y_test_standard = results_df["actual"]
y_pred_values = results_df["predicted"]

# get scatter plot of actual vs predicted for thesis plots with histograms
g = sns.jointplot(x=y_test_standard, y=y_pred_values, kind="scatter", s=1)

# Add line corresponding to perfect prediction
plt.plot([0.0, 1.0], [0.0, 1.0], color="red")

# Set labels of axis
plt.xlabel("True GPP")
plt.ylabel("Predicted GPP")

# Save plot
plt.savefig('de_hai_xgb.png')