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


# Define function creating the sequences so that it has the input format for the LSTM
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
        y_seq.append(y[i + seq_length])  # predict the last time step
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
    unique_sites = df[site_col].unique()
    unique_sites = sorted(unique_sites)

    if site_arg.isdigit():
        idx = int(site_arg)
        this_site = unique_sites[idx]
    else:
        this_site = site_arg

    # 5) Split data: train = all sites except this_site, test = this_site
    test_mask = df[site_col] == this_site
    train_mask = ~test_mask

    df_train = df.loc[train_mask].copy()
    df_test = df.loc[test_mask].copy()

    # Edge case: no data for this site
    if df_test.shape[0] == 0:
        print(f"Site {this_site} has no rows. Skipping.")
        return

    X_train_raw = df_train[features].values
    y_train_raw = df_train[target_col].values
    X_test_raw = df_test[features].values
    y_test_raw = df_test[target_col].values

    # 6) Scale X
    x_scaler = MinMaxScaler()
    X_train_scaled = x_scaler.fit_transform(X_train_raw)
    X_test_scaled = x_scaler.transform(X_test_raw)

    # 7) Scale y (manually or with MinMaxScaler)
    y_min, y_max = y_train_raw.min(), y_train_raw.max()
    y_train_scaled = (y_train_raw - y_min) / (y_max - y_min)
    y_test_scaled = (y_test_raw - y_min) / (y_max - y_min)

    # 8) Reshape data for LSTM
    #    LSTM expects shape = (samples, timesteps, features)
    #    This depends heavily on how your time dimension is structured.
    #    For a quick example, let's assume each row is a single time step,
    #    and we want to use a window size = seq_length timesteps.
    
    seq_length = 10  # example: 10 time steps per sequence
    X_train_seq, y_train_seq = create_sequences(X_train_scaled, y_train_scaled, seq_length)
    X_test_seq,  y_test_seq  = create_sequences(X_test_scaled,  y_test_scaled,  seq_length)

    # 9) Build LSTM model
    model = Sequential()
    model.add(LSTM(units=64, input_shape=(seq_length, X_train_seq.shape[2])))
    model.add(Dense(1))  # single output for regression

    model.compile(optimizer='adam', loss='mse')
    
    # Consider early stopping for not improving loss
    early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

    # 10) Train
    model.fit(
        X_train_seq, y_train_seq,
        epochs=10,
        batch_size=32,
        validation_split=0.1,
        verbose=0,
        callbacks=[early_stopping]
    )

    # 11) Predict
    y_pred_scaled = model.predict(X_test_seq).flatten()

    # 12) Evaluate metrics in scaled space
    mse_scaled = mean_squared_error(y_test_seq, y_pred_scaled)
    rmse_scaled = np.sqrt(mse_scaled)
    r2_scaled = r2_score(y_test_seq, y_pred_scaled)
    mae_scaled = np.mean(np.abs(y_test_seq - y_pred_scaled))
    relative_error = np.mean(np.abs(y_test_seq - y_pred_scaled) / np.abs(y_test_seq))

    # 13) Save results
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
    # site_arg could be e.g. "0" or "SITE123". We'll handle both
    if len(sys.argv) < 2:
        print("Usage: train_one_site.py <site_arg>")
        sys.exit(1)

    site_arg = sys.argv[1]
    main(site_arg)
