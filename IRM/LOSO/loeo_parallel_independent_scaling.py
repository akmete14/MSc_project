import pandas as pd
import torch
from torch.autograd import grad
import argparse
import numpy as np
import os

print("modules and torch imported")

############################
### MAKE DATAFRAME READY ###
############################

# Read the CSV file into a DataFrame and drop not necessariy columns 'Unnamed: 0' and 'cluster'
df = pd.read_csv('/cluster/project/math/akmete/MSc/preprocessing/df_balanced_groups_nonans.csv')
df = df.drop(columns=['Unnamed: 0', 'cluster'])

print(df.columns)
print(df.head())
print(len(df))

# Identify feature columns (all columns except 'GPP' and 'site_id')
feature_columns = [col for col in df.columns if col not in ['GPP', 'site_id']]
target_column = "GPP"

# Prepare the environments list (each environment corresponds to a site)
environments = []
# Group the DataFrame by 'site_id'
for site, group in df.groupby('site_id'):
    # Convert features and target to PyTorch tensors
    X = torch.tensor(group[feature_columns].values, dtype=torch.float32)
    y = torch.tensor(group['GPP'].values, dtype=torch.float32).view(-1, 1)
    # Each environment is a tuple (X, y)
    environments.append((X, y))
    print(f"Site {site}: X shape = {X.shape}, y shape = {y.shape}")


################################
### DEFINE SCALING FUNCTIONS ###
################################

def scale_environment(environment):
    """
    Scales a single environment (tuple of tensors: (X, y)) using min-max scaling.
    """
    x, y = environment
    # Compute min and max for features in this environment
    min_x, _ = torch.min(x, dim=0)
    max_x, _ = torch.max(x, dim=0)
    # Compute min and max for the target in this environment
    min_y, _ = torch.min(y, dim=0)
    max_y, _ = torch.max(y, dim=0)

    # Avoid division by zero by replacing zeros in the difference with 1
    diff_x = max_x - min_x
    diff_x[diff_x == 0] = 1

    # Scale the features and target
    scaled_x = (x - min_x) / diff_x
    scaled_y = (y - min_y) / (max_y - min_y + 1e-8)

    return (scaled_x, scaled_y)

def scale_environments_independent(train_environments, test_environments):
    # Scale each training environment independently
    scaled_train_envs = [scale_environment(env) for env in train_environments]
    # Scale each test environment independently
    scaled_test_envs = [scale_environment(env) for env in test_environments]
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

# Define training parameters
args = {
    "lr": 0.01,
    "n_iterations": 5000,
    "verbose": False  # Turn off verbose logging for cluster jobs
}

# Use the random sites as site_ids for folds
site_ids = df['site_id'].unique()[:len(environments)]

def run_fold(fold_index):
    # Leave out the fold specified by fold_index
    train_environments = [env for j, env in enumerate(environments) if j != fold_index]
    test_environment = [environments[fold_index]]
    
    # Scale the data
    scaled_train_envs, scaled_test_envs = scale_environments_independent(train_environments, test_environment)
    
    # Train the model on the training environments
    irm_model = IRM(scaled_train_envs, args)
    
    # Evaluate on the left-out test environment
    x_test, y_test = scaled_test_envs[0]
    y_pred_scaled = x_test @ irm_model.solution()
    test_mse = ((y_pred_scaled - y_test) ** 2).mean().item()
    test_rmse = np.sqrt(test_mse)
    
    print(f"Fold {fold_index}: Site {site_ids[fold_index]} - Test MSE: {test_mse:.5f}, Test RMSE: {test_rmse:.5f}")
    
    # Save results to a CSV file. Each job writes its own CSV file.
    results_df = pd.DataFrame([{
        'site_id': site_ids[fold_index],
        'mse': test_mse,
        'rmse': test_rmse
    }])
    
    # Create an output directory if it doesn't exist
    os.makedirs("results_independent_scaling", exist_ok=True)
    output_filename = os.path.join("results_independent_scaling", f"fold_{fold_index}_results.csv")
    results_df.to_csv(output_filename, index=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run a single LOEO fold.")
    parser.add_argument("fold", type=int, help="Index of the fold to leave out (0-9)")
    args_parsed = parser.parse_args()
    
    run_fold(args_parsed.fold)