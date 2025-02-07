import pandas as pd
import torch
from torch.autograd import grad
import argparse
import numpy as np
import multiprocessing as mp
print("modules and torch imported")

############################
### MAKE DATAFRAME READY ###
############################

df = pd.read_csv('/cluster/project/math/akmete/MSc/preprocessing/df_balanced_groups_nonans.csv')

# Get the first 10 unique site IDs
first_10_sites = df['site_id'].unique()[:10]

# Filter the DataFrame for these site IDs
subset_df = df[df['site_id'].isin(first_10_sites)]
df = subset_df.drop(columns=['Unnamed: 0', 'cluster'])

print(df.columns)
print(df.head())
print(len(df))

# Identify feature columns
feature_columns = [col for col in df.columns if col not in ['GPP', 'site_id']]
target_column = "GPP"

# Prepare the environments list (environments correspond to different sites)
environments = []
# Group the DataFrame by 'site_id'
for site, group in df.groupby('site_id'):
    # Convert features and target to PyTorch tensors
    X = torch.tensor(group[feature_columns].values, dtype=torch.float32)
    y = torch.tensor(group['GPP'].values, dtype=torch.float32).view(-1, 1)
    # Each environment is a tuple (X, y)
    environments.append((X, y))
    print(f"Site {site}: X shape = {X.shape}, y shape = {y.shape}")


# Define scaling #
# Define Min-Max Scaling (X and Y)
def scale_environments(train_environments, test_environments):
    train_x = torch.cat([env[0] for env in train_environments], dim=0)
    train_y = torch.cat([env[1] for env in train_environments], dim=0)

    min_x, _ = torch.min(train_x, dim=0)
    max_x, _ = torch.max(train_x, dim=0)
    min_y, _ = torch.min(train_y, dim=0)
    max_y, _ = torch.max(train_y, dim=0)

    def minmax_scale_x(x):
        diff = max_x - min_x
        diff[diff == 0] = 1  # Avoid division by zero
        return (x - min_x) / diff
        #return (x - min_x) / (max_x - min_x + 1e-8)

    def minmax_scale_y(y):
        return (y - min_y) / (max_y - min_y + 1e-8)

    scaled_train_envs = [(minmax_scale_x(x), minmax_scale_y(y)) for x, y in train_environments]
    scaled_test_envs = [(minmax_scale_x(x), minmax_scale_y(y)) for x, y in test_environments]

    return scaled_train_envs, scaled_test_envs





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

    def train(self, environments, args, reg=0):
        dim_x = environments[0][0].size(1)
        self.phi = torch.nn.Parameter(torch.eye(dim_x, dim_x))
        self.w = torch.ones(dim_x, 1, requires_grad=True)

        opt = torch.optim.Adam([self.phi], lr=args["lr"])
        loss = torch.nn.MSELoss()

        for iteration in range(args["n_iterations"]):
            penalty = 0
            error = 0
            for x_e, y_e in environments:
                error_e = loss(x_e @ self.phi @ self.w, y_e)
                penalty += grad(error_e, self.w, create_graph=True)[0].pow(2).mean()
                error += error_e

            opt.zero_grad()
            (reg * error + (1 - reg) * penalty).backward()
            opt.step()

            if args["verbose"] and iteration % 1000 == 0:
                print(f"Iteration {iteration:05d} | Reg: {reg:.5f} | Error: {error:.5f} | Penalty: {penalty:.5f}")

    def solution(self):
        return (self.phi @ self.w).view(-1, 1)



################
### LOEO-CV  ###
################

site_ids = df['site_id'].unique()[:len(environments)]

# Define LOEO function
def run_loeo_fold(i, environments, site_ids, args):
    # Select train and test environments for fold i
    train_environments = [env for j, env in enumerate(environments) if j != i]
    test_environment = [environments[i]]
    
    # Scale the data using your scaling function
    scaled_train_envs, scaled_test_envs = scale_environments(train_environments, test_environment)
    
    # Train the model on the training environments
    irm_model = IRM(scaled_train_envs, args)
    
    # Evaluate on the left-out test environment
    x_test, y_test = scaled_test_envs[0]
    y_pred_scaled = x_test @ irm_model.solution()
    test_mse = ((y_pred_scaled - y_test) ** 2).mean().item()
    test_rmse = np.sqrt(test_mse)
    
    print(f"Fold {i}: Site {site_ids[i]} - Test MSE: {test_mse:.5f}, Test RMSE: {test_rmse:.5f}")
    return (site_ids[i], test_mse, test_rmse)


if __name__ == '__main__':
    # Define parameters for training
    args = {
        "lr": 0.01,           # Learning rate
        "n_iterations": 5000, # Training iterations
        "verbose": False      # Turn off verbose printing to avoid clutter in parallel output
    }
    
    # Assume `environments` and `site_ids` have already been defined as in your original code.
    site_ids = df['site_id'].unique()[:len(environments)]
    
    # Create a multiprocessing Pool with as many processes as there are CPU cores
    pool = mp.Pool(processes=mp.cpu_count())
    
    # Prepare the list of arguments for each fold
    fold_args = [(i, environments, site_ids, args) for i in range(len(environments))]
    
    # Run LOEO folds in parallel using starmap to unpack the tuple arguments
    results = pool.starmap(run_loeo_fold, fold_args)
    
    # Clean up the pool
    pool.close()
    pool.join()
    
    # Collect and save results
    results_df = pd.DataFrame(results, columns=['site_id', 'mse', 'rmse'])
    results_df.to_csv('Leave-One-Environment-Out_10sites.csv', index=False)
