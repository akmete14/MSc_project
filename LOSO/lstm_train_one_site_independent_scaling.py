### Revised by gpt, check again before running (compare with lstm_train_one_site.py) ###

import sys
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler

# Using TensorFlow/Keras
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

# Force TensorFlow to use 8 threads for both inter_op and intra_op
tf.config.threading.set_intra_op_parallelism_threads(8)
tf.config.threading.set_inter_op_parallelism_threads(8)


# Define function to create sequences for the LSTM
def create_sequences(X, y, seq_length=10):
    """
    Given X (samples, features) and y (samples,),
    return LSTM-ready arrays: (num_sequences, seq_length, features)
    and (num_sequences,) or (num_sequences, 1).
    Assumes data is already time-ordered!
    """
    X_seq, y_seq = [], []
    for i in range(len(X) - seq_length):
        X_seq.append(X[i : i+seq_length, :])
        y_seq.append(y[i + seq_length])  # predict the value after the sequence
    return np.array(X_seq), np.array(y_seq)


def main(site_arg):

    # 1) Load data
    df = pd.read_csv('/cluster/project/math/akmete/MSc/preprocessing/df_balanced_groups_nonans.csv')

    # 2) Convert float64 to float32 for memory savings
    for col in df.select_dtypes(include=["float64"]).columns:
        df[col] = df[col].astype('float32')

    # 3) Identify columns
    site_col = "site_id"   # Identifies each site
    target_col = "GPP"     # The column we want to predict

    # Non-feature columns
    non_features = [target_col, site_col, "cluster"]
    features = [c for c in df.columns if c not in non_features]

    # 4) Convert site_arg to site ID if needed
    unique_sites = sorted(df[site_col].unique())
    if site_arg.isdigit():
        idx = int(site_arg)
        this_site = unique_sites[idx]
    else:
        this_site = site_arg

    # 5) Split data: training = all sites except this_site, testing = this_site
    test_mask = df[site_col] == this_site
    train_mask = ~test_mask

    df_train = df.loc[train_mask].copy()
    df_test = df.loc[test_mask].copy()

    # Edge case: no data for this site
    if df_test.shape[0] == 0:
        print(f"Site {this_site} has no rows. Skipping.")
        return

    # -------------------------------
    # 6) Independent Scaling for Training Data
    # -------------------------------
    # For each site in the training set, scale features and target independently.
    scaled_train_dfs = []
    for site, group in df_train.groupby(site_col):
        # Scale X for this site
        scaler_x = MinMaxScaler()
        X_site_scaled = scaler_x.fit_transform(group[features].values)
        
        # Scale y for this site manually
        y_site = group[target_col].values
        y_min, y_max = y_site.min(), y_site.max()
        if y_max == y_min:
            y_site_scaled = np.zeros_like(y_site)
        else:
            y_site_scaled = (y_site - y_min) / (y_max - y_min)
        
        # Create a DataFrame with scaled features and target
        group_scaled = pd.DataFrame(X_site_scaled, columns=features, index=group.index)
        group_scaled[target_col] = y_site_scaled
        scaled_train_dfs.append(group_scaled)
    
    # Concatenate the scaled training data from all sites
    df_train_scaled = pd.concat(scaled_train_dfs)
    # Ensure the ordering is the same as in the original training set
    df_train_scaled = df_train_scaled.loc[df_train.index]

    X_train_scaled = df_train_scaled[features].values
    y_train_scaled = df_train_scaled[target_col].values

    # -------------------------------
    # 7) Independent Scaling for Test Data (left-out site)
    # -------------------------------
    # For the test set (which is a single site), fit a scaler on its own data.
    scaler_test = MinMaxScaler()
    X_test_scaled = scaler_test.fit_transform(df_test[features].values)
    
    y_test_raw = df_test[target_col].values
    y_min_test, y_max_test = y_test_raw.min(), y_test_raw.max()
    if y_max_test == y_min_test:
        y_test_scaled = np.zeros_like(y_test_raw)
    else:
        y_test_scaled = (y_test_raw - y_min_test) / (y_max_test - y_min_test)

    # -------------------------------
    # 8) Reshape data for LSTM
    # -------------------------------
    # LSTM expects data shape = (samples, timesteps, features)
    seq_length = 10  # example: use 10 time steps per sequence
    X_train_seq, y_train_seq = create_sequences(X_train_scaled, y_train_scaled, seq_length)
    X_test_seq,  y_test_seq  = create_sequences(X_test_scaled,  y_test_scaled,  seq_length)

    # -------------------------------
    # 9) Build LSTM Model
    # -------------------------------
    model = Sequential()
    model.add(LSTM(units=64, input_shape=(seq_length, X_train_seq.shape[2])))
    model.add(Dense(1))  # Single output for regression
    model.compile(optimizer='adam', loss='mse')
    
    early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

    # -------------------------------
    # 10) Train
    # -------------------------------
    model.fit(
        X_train_seq, y_train_seq,
        epochs=10,
        batch_size=32,
        validation_split=0.1,
        verbose=0,
        callbacks=[early_stopping]
    )

    # -------------------------------
    # 11) Predict
    # -------------------------------
    y_pred_scaled = model.predict(X_test_seq).flatten()

    # -------------------------------
    # 12) Evaluate Metrics in Scaled Space
    # -------------------------------
    mse_scaled = mean_squared_error(y_test_seq, y_pred_scaled)
    rmse_scaled = np.sqrt(mse_scaled)
    r2_scaled = r2_score(y_test_seq, y_pred_scaled)
    mae_scaled = np.mean(np.abs(y_test_seq - y_pred_scaled))
    relative_error = np.mean(np.abs(y_test_seq - y_pred_scaled) / np.abs(y_test_seq))

    # -------------------------------
    # 13) Save Results
    # -------------------------------
    results = {
        "site_left_out": this_site,
        "mse_scaled": mse_scaled,
        "rmse_scaled": rmse_scaled,
        "r2_scaled": r2_scaled,
        "mae_scaled": mae_scaled,
        "relative_error": relative_error
    }
    out_df = pd.DataFrame([results])
    out_df.to_csv(f"results_{this_site}_lstm.csv", index=False)
    print(f"Done with site {this_site}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: train_one_site.py <site_arg>")
        sys.exit(1)
    site_arg = sys.argv[1]
    main(site_arg)
