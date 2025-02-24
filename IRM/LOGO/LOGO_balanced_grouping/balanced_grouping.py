### LOGO with balanced grouping ###

import pandas as pd
import torch
from torch.autograd import grad
import argparse
import numpy as np
import os
from tqdm import tqdm

print("modules and torch imported")

############################
### MAKE DATAFRAME READY ###
############################

# Read the CSV file into a DataFrame and drop not necessariy column 'Unnamed: 0'
df = pd.read_csv('/cluster/project/math/akmete/MSc/preprocessing/df_balanced_groups_onevegindex.csv')
df = df.dropna(axis=1, how='all')  # Drop columns where all values are NaN
df = df.fillna(0)
df = df.drop(columns=['Unnamed: 0'])

for col in tqdm(df.select_dtypes(include=['float64']).columns, desc="Casting columns"):
    df[col] = df[col].astype('float32')

print(df.columns)
print(df.head())
print(len(df))

# Identify feature columns (all columns except 'GPP', 'cluster' and 'site_id')
feature_columns = [col for col in df.columns if col not in ['GPP', 'site_id','cluster']]
target_column = "GPP"

# Create a dictionary to hold a list of site-level environments for each cluster
environments_by_cluster = {}
for cluster, cluster_df in df.groupby('cluster'):
    site_envs = []
    for site, site_df in cluster_df.groupby('site_id'):
        X = torch.tensor(site_df[feature_columns].values, dtype=torch.float32)
        y = torch.tensor(site_df[target_column].values, dtype=torch.float32).view(-1, 1)
        site_envs.append((X, y))
        print(f"Cluster {cluster}, Site {site}: X shape = {X.shape}, y shape = {y.shape}")
    environments_by_cluster[cluster] = site_envs


################################
### DEFINE SCALING FUNCTIONS ###
################################

def compute_global_scaling_stats(environments):
    """
    Given a list of environments (tuples of tensors: (X, y)), computes the global
    minimum and maximum for the features and target across all environments.
    """
    # Concatenate all X and y from the environments
    X_all = torch.cat([env[0] for env in environments], dim=0)
    y_all = torch.cat([env[1] for env in environments], dim=0)
    
    # Compute global min and max for features
    global_min_x, _ = torch.min(X_all, dim=0)
    global_max_x, _ = torch.max(X_all, dim=0)
    
    # Compute global min and max for the target
    global_min_y, _ = torch.min(y_all, dim=0)
    global_max_y, _ = torch.max(y_all, dim=0)
    
    return {
        "min_x": global_min_x,
        "max_x": global_max_x,
        "min_y": global_min_y,
        "max_y": global_max_y,
    }

def scale_environment_global(environment, global_stats):
    """
    Scales a single environment (tuple of tensors: (X, y)) using the provided
    global statistics.
    """
    x, y = environment
    min_x = global_stats["min_x"]
    max_x = global_stats["max_x"]
    min_y = global_stats["min_y"]
    max_y = global_stats["max_y"]
    
    # Avoid division by zero for features
    diff_x = max_x - min_x
    diff_x[diff_x == 0] = 1.0

    scaled_x = (x - min_x) / diff_x
    scaled_y = (y - min_y) / (max_y - min_y + 1e-8)
    
    return (scaled_x, scaled_y)

#####################
### USE IRM CLASS ###
#####################

class IRM:
    def __init__(self, environments, args):
        best_reg = 0
        best_err = 1e6
        x_val = environments[-1][0]  
        y_val = environments[-1][1]

        for reg in [0, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]:
            self.train(environments[:-1], args, reg=reg)
            err = (x_val @ self.solution() - y_val).pow(2).mean().item()

            if args["verbose"]:
                print(f"IRM (reg={reg:.3f}) has {err:.3f} validation error.")

            if err < best_err:
                best_err = err
                best_reg = reg
                best_phi = self.phi.clone()
        self.phi = best_phi

    # Updated train so that it will do early stopping when not improving
    def train(self, environments, args, reg=0):
        dim_x = environments[0][0].size(1)
        # Initialize phi as an identity matrix and w as ones
        self.phi = torch.nn.Parameter(torch.eye(dim_x, dim_x))
        self.w = torch.ones(dim_x, 1, requires_grad=True)

        opt = torch.optim.Adam([self.phi], lr=args["lr"])
        loss_fn = torch.nn.MSELoss()

        # Early stopping setup
        best_loss = float('inf')
        patience_counter = 0
        patience = args.get("early_stopping_patience", 100)
        min_delta = args.get("early_stopping_min_delta", 1e-4)

        for iteration in range(args["n_iterations"]):
            penalty = 0
            error = 0
            for x_e, y_e in environments:
                # Compute the error for this environment
                error_e = loss_fn(x_e @ self.phi @ self.w, y_e)
                # Compute the penalty term using the gradient of the error w.r.t. w
                penalty += grad(error_e, self.w, create_graph=True)[0].pow(2).mean()
                error += error_e

            # Calculate the combined loss
            total_loss = reg * error + (1 - reg) * penalty

            # Backpropagation
            opt.zero_grad()
            total_loss.backward()
            opt.step()

            current_loss = total_loss.item()

            # Check for improvement
            if current_loss < best_loss - min_delta:
                best_loss = current_loss
                patience_counter = 0
            else:
                patience_counter += 1

            if args["verbose"] and iteration % 1000 == 0:
                print(f"Iteration {iteration:05d} | Reg: {reg:.5f} | Error: {error:.5f} | "
                    f"Penalty: {penalty:.5f} | Total Loss: {current_loss:.5f}")

            # Early stopping: break if no improvement for 'patience' iterations
            if patience_counter >= patience:
                if args["verbose"]:
                    print(f"Early stopping at iteration {iteration} with best total loss {best_loss:.5f}")
                break

    def solution(self):
        return (self.phi @ self.w).view(-1, 1)


################
### LOEO-CV  ###
################

# Define training parameters
args = {
    "lr": 0.01,
    "n_iterations": 1000,
    "verbose": False,  # Turn off verbose logging for cluster jobs
    "early_stopping_patience": 100,  # stop if no improvement for 100 iterations
    "early_stopping_min_delta": 1e-4
}

# Get list of cluster IDs
cluster_ids = list(environments_by_cluster.keys())

def run_fold(fold_cluster):
    # Collect training and test environments from clusters
    train_envs = []
    test_envs = []
    for cluster, envs in environments_by_cluster.items():
        if cluster == fold_cluster:
            test_envs.extend(envs)
        else:
            train_envs.extend(envs)
    
    # Check if the test environments for the selected cluster exist
    if len(test_envs) == 0:
        raise ValueError(f"No test environments found for fold_cluster: {fold_cluster}")

    # Compute global scaling statistics using only the training environments
    global_stats = compute_global_scaling_stats(train_envs)
    
    # Scale training and test environments using the same global statistics
    scaled_train_envs = [scale_environment_global(env, global_stats) for env in train_envs]
    scaled_test_envs = [scale_environment_global(env, global_stats) for env in test_envs]
    
    # Ensure that there is at least one scaled test environment before concatenation
    if not scaled_test_envs or len(scaled_test_envs) == 0:
        raise ValueError(f"After scaling, no test environments available for fold_cluster: {fold_cluster}")
    
    # Train IRM on training environments (note: the IRM class uses the last env for validation)
    irm_model = IRM(scaled_train_envs, args)
    
    # Combine test environments (sites) for evaluation
    X_test = torch.cat([env[0] for env in scaled_test_envs], dim=0)
    y_test = torch.cat([env[1] for env in scaled_test_envs], dim=0)
    y_pred_scaled = X_test @ irm_model.solution()
    test_mse = ((y_pred_scaled - y_test) ** 2).mean().item()
    test_rmse = np.sqrt(test_mse)
    # Calculate r2, relative error and mae
    ss_res = ((y_test - y_pred_scaled) ** 2).sum()
    ss_tot = ((y_test - y_test.mean()) ** 2).sum()
    r2 = (1 - ss_res / ss_tot).item()
    # Mean Absolute Error (MAE)
    mae = torch.abs(y_test - y_pred_scaled).mean().item()
    # Relative error: average of abs((y_pred - y_true)/y_true)
    relative_error = (torch.abs(y_test - y_pred_scaled) / torch.abs(y_test)).mean().item()
    
    print(f"Cluster {fold_cluster}: Test MSE: {test_mse:.5f}, Test RMSE: {test_rmse:.5f}, R2 Score: {r2:.5f}, MAE: {mae:.5f}, Relative Error: {relative_error:.5f}")
    
    # Save results to CSV (each fold gets its own file)
    results_df = pd.DataFrame([{
        'cluster': fold_cluster,
        'mse': test_mse,
        'rmse': test_rmse,
        'r2': r2,
        'relative_error': relative_error,
        'mae': mae
    }])
    os.makedirs("LOGO_balanced_grouping", exist_ok=True)
    output_filename = os.path.join("LOGO_balanced_grouping", f"cluster_{fold_cluster}_results.csv")
    results_df.to_csv(output_filename, index=False)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run a single LOGO fold by cluster.")
    # You can either pass the actual cluster id or an index mapping to cluster_ids
    parser.add_argument("fold_cluster", type=int, help="Cluster id to leave out")
    args_parsed = parser.parse_args()
    
    run_fold(args_parsed.fold_cluster)

