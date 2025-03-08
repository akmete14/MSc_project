import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tqdm import tqdm

# -----------------------
# Load and Preprocess Data
# -----------------------
df = pd.read_csv('/cluster/project/math/akmete/MSc/preprocessing/df_balanced_groups_onevegindex.csv')
df = df.dropna(axis=1, how='all')  # Drop columns where all values are NaN
df = df.fillna(0)
df = df.drop(columns=['Unnamed: 0','cluster'])
for col in tqdm(df.select_dtypes(include=['float64']).columns, desc="Casting columns"):
    df[col] = df[col].astype('float32')

# Define features and target
feature_columns = [col for col in df.columns if col not in ['GPP', 'site_id']]
target_column = "GPP"

# -----------------------
# Create Sequences Function
# -----------------------
def create_sequences(X, y, seq_len=10):
    """
    Build sequences of shape (num_sequences, seq_len, num_features) for X
    and corresponding targets (num_sequences,) for y.
    """
    if isinstance(X, (pd.DataFrame, pd.Series)):
        X = X.to_numpy()
    if isinstance(y, (pd.DataFrame, pd.Series)):
        y = y.to_numpy()
        
    X_seq, y_seq = [], []
    for i in range(len(X) - seq_len):
        X_seq.append(X[i:i+seq_len])
        y_seq.append(y[i+seq_len])
    return np.array(X_seq), np.array(y_seq)

# -----------------------
# Process only the site 'DE-Hai'
# -----------------------
target_site = "DE-Hai"
df_site = df[df['site_id'] == target_site].copy()
print(f"Processing site {target_site}: shape {df_site.shape}")

# Drop any remaining rows with missing values
df_site = df_site.dropna(axis=1, how='all').dropna()

# Chronological 80/20 split
split_index = int(len(df_site) * 0.8)
df_train = df_site.iloc[:split_index]
df_test  = df_site.iloc[split_index:]
print(f"Site {target_site}: Training samples: {len(df_train)}, Testing samples: {len(df_test)}")

# Separate features and target
X_train_time_raw = df_train[feature_columns]
y_train_time_raw = df_train[target_column]
X_test_time_raw  = df_test[feature_columns]
y_test_time_raw  = df_test[target_column]

# Scale target using training statistics (min–max scaling)
y_train_standard = (y_train_time_raw - y_train_time_raw.min()) / (y_train_time_raw.max() - y_train_time_raw.min())
y_test_standard  = (y_test_time_raw - y_train_time_raw.min()) / (y_train_time_raw.max() - y_train_time_raw.min())

# Scale features using MinMaxScaler (fit on training data)
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train_time_raw)
X_test_scaled  = scaler.transform(X_test_time_raw)

# Create sequences (LSTM requires 3D input)
seq_len = 10
X_train_seq, y_train_seq = create_sequences(X_train_scaled, y_train_standard, seq_len)
X_test_seq,  y_test_seq  = create_sequences(X_test_scaled,  y_test_standard, seq_len)

print(f"Site {target_site}: X_train_seq shape: {X_train_seq.shape}, y_train_seq shape: {y_train_seq.shape}")
print(f"Site {target_site}: X_test_seq shape: {X_test_seq.shape}, y_test_seq shape: {y_test_seq.shape}")

# -----------------------
# Build and Train LSTM Model
# -----------------------
model = Sequential()
model.add(LSTM(64, input_shape=(seq_len, X_train_seq.shape[2])))
model.add(Dense(1))  # Single-output regression
model.compile(optimizer='adam', loss='mse')
model.summary()

# Train the model (using 10% of training data as validation)
history = model.fit(
    X_train_seq, y_train_seq,
    validation_split=0.1,
    epochs=20,
    batch_size=32,
    verbose=0
)

# Evaluate on the test set
test_loss = model.evaluate(X_test_seq, y_test_seq, verbose=0)

# Get predictions and compute additional metrics
y_pred = model.predict(X_test_seq).flatten()
mse = mean_squared_error(y_test_seq, y_pred)
r2 = r2_score(y_test_seq, y_pred)
relative_error = np.mean(np.abs(y_test_seq - y_pred) / np.abs(y_test_seq))
mae = mean_absolute_error(y_test_seq, y_pred)
rmse = np.sqrt(mse)

print(f"Metrics for site {target_site}: Loss={test_loss:.6f}, MSE={mse:.6f}, R2={r2:.6f}, Relative Error={relative_error:.6f}, MAE={mae:.6f}, RMSE={rmse:.6f}")

# -----------------------
# Save Results and Predictions
# -----------------------
# Save overall performance metrics to a CSV file
results_df = pd.DataFrame({
    'site': [target_site],
    'test_loss': [test_loss],
    'mse': [mse],
    'r2': [r2],
    'relative_error': [relative_error],
    'mae': [mae],
    'rmse': [rmse]
})
results_filename = f"results_{target_site}.csv"
results_df.to_csv(results_filename, index=False)
print(f"Overall results saved to {results_filename}")

# Save per-sample predictions to a CSV file (for overlayed scatter plot later)
predictions_df = pd.DataFrame({
    'site': target_site,
    'method': 'LSTM',
    'actual': y_test_seq,
    'predicted': y_pred
})
predictions_filename = f"predictions_{target_site}.csv"
predictions_df.to_csv(predictions_filename, index=False)
print(f"Predictions saved to {predictions_filename}")
