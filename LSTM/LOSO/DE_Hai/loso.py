import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------
# 1) Load and Preprocess Data
# -----------------------
df = pd.read_csv('/cluster/project/math/akmete/MSc/preprocessing/df_balanced_groups_onevegindex.csv')
df = df.dropna(axis=1, how='all')  # Drop columns where all values are NaN
df = df.fillna(0)
df = df.drop(columns=['Unnamed: 0', 'cluster'])
for col in tqdm(df.select_dtypes(include=['float64']).columns, desc="Casting columns"):
    df[col] = df[col].astype('float32')

# Define features and target
feature_columns = [col for col in df.columns if col not in ['GPP', 'site_id']]
target_column = "GPP"

# -----------------------
# 2) Sequence Generation Functions
# -----------------------
def sequence_generator(X, y, seq_len=10, batch_size=32):
    """
    Yields batches of sequences for X and corresponding targets for y.
    """
    X = np.asarray(X)
    y = np.asarray(y)
    total = len(X) - seq_len
    while True:
        for i in range(0, total, batch_size):
            X_batch = []
            y_batch = []
            for j in range(i, min(i + batch_size, total)):
                X_batch.append(X[j:j+seq_len])
                y_batch.append(y[j+seq_len])
            yield np.array(X_batch), np.array(y_batch)

def create_sequences(X, y, seq_len=10):
    """
    Build sequences of shape (num_sequences, seq_len, num_features) for X
    and corresponding targets for y.
    """
    X = np.asarray(X)
    y = np.asarray(y)
    X_seq, y_seq = [], []
    for i in range(len(X) - seq_len):
        X_seq.append(X[i:i+seq_len])
        y_seq.append(y[i+seq_len])
    return np.array(X_seq), np.array(y_seq)

# -----------------------
# 3) Define the 'DE-Hai' Fold
# -----------------------
test_site = 'DE-Hai'
df_train = df[df['site_id'] != test_site].copy()
df_test  = df[df['site_id'] == test_site].copy()

# -----------------------
# 4) Extract Features and Target and Scale Them
# -----------------------
X_train_raw = df_train[feature_columns]
y_train_raw = df_train[target_column]
X_test_raw  = df_test[feature_columns]
y_test_raw  = df_test[target_column]

# Scale target variable based on training data (min-max scaling)
y_train_min = y_train_raw.min()
y_train_max = y_train_raw.max()
if y_train_max - y_train_min == 0:
    y_train_scaled_vals = y_train_raw.to_numpy()
    y_test_scaled_vals  = y_test_raw.to_numpy()
else:
    y_train_scaled_vals = ((y_train_raw - y_train_min) / (y_train_max - y_train_min)).to_numpy()
    y_test_scaled_vals  = ((y_test_raw - y_train_min) / (y_train_max - y_train_min)).to_numpy()

# Scale features using MinMaxScaler (fit on training data)
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train_raw)
X_test_scaled  = scaler.transform(X_test_raw)

# -----------------------
# 5) Prepare Sequences for LSTM
# -----------------------
seq_len = 10
batch_size = 32

# Training generator (for batch training)
train_gen = sequence_generator(X_train_scaled, y_train_scaled_vals, seq_len=seq_len, batch_size=batch_size)
steps_per_epoch = (len(X_train_scaled) - seq_len) // batch_size

# For testing, create all sequences in memory (assumes test data is small)
X_test_seq, y_test_seq = create_sequences(X_test_scaled, y_test_scaled_vals, seq_len)

print(f"Training steps per epoch: {steps_per_epoch}")
print(f"X_test_seq shape: {X_test_seq.shape}, y_test_seq shape: {y_test_seq.shape}")

# -----------------------
# 6) Build and Train the LSTM Model
# -----------------------
model = Sequential()
model.add(LSTM(64, input_shape=(seq_len, X_train_scaled.shape[1])))
model.add(Dense(1))  # Single-output regression
model.compile(optimizer='adam', loss='mse')
model.summary()

history = model.fit(
    train_gen,
    steps_per_epoch=steps_per_epoch,
    validation_data=(X_test_seq, y_test_seq),
    epochs=20,
    verbose=1
)

# -----------------------
# 7) Evaluate the Model and Compute Metrics
# -----------------------
test_loss = model.evaluate(X_test_seq, y_test_seq, verbose=0)

y_pred = model.predict(X_test_seq).flatten()
mse_val = mean_squared_error(y_test_seq, y_pred)
r2_val = r2_score(y_test_seq, y_pred)
relative_error = np.mean(np.abs(y_test_seq - y_pred) / np.abs(y_test_seq))
mae_val = mean_absolute_error(y_test_seq, y_pred)
rmse_val = np.sqrt(mse_val)

print(f"Metrics for held-out site {test_site}:")
print(f"Loss={test_loss:.6f}, MSE={mse_val:.6f}, R2={r2_val:.6f}, Relative Error={relative_error:.6f}, MAE={mae_val:.6f}, RMSE={rmse_val:.6f}")

# -----------------------
# 8) Save Actual and Predicted Values
# -----------------------
results_df = pd.DataFrame({
    'actual': y_test_seq,
    'predicted': y_pred
})
results_df.to_csv("results_DE-Hai_lstm.csv", index=False)
print("Results saved to results_DE-Hai_lstm.csv")

# -----------------------
# 9) Joint Scatter Plot with Marginal Histograms
# -----------------------
g = sns.jointplot(x=results_df["actual"], y=results_df["predicted"], kind="scatter", s=1)
plt.plot([0.0, 1.0], [0.0, 1.0], color="red")  # y=x reference line
plt.xlabel("True GPP")
plt.ylabel("Predicted GPP")
plt.savefig('de_hai_lstm.png')
