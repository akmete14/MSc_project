import pandas as pd
import torch
from torch.autograd import grad
import argparse
import numpy
print("modules and torch imported")

############################
### MAKE DATAFRAME READY ###
############################

df = pd.read_csv('/cluster/project/math/akmete/MSc/preprocessing/df_balanced_groups_nonans.csv')

# Get the first 5 unique site IDs
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
import torch
from torch.autograd import grad

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



#######################
### TRAIN IRM MODEL ###
#######################

args = {
    "lr": 0.01,           # Learning rate
    "n_iterations": 5000, # Training iterations
    "verbose": True       # Print updates every 1000 iterations
}

# Split environments into training and test
train_environments = environments[:-1]  # First N-1 environments for training
test_environments = [environments[-1]]  # Last environment for testing

# Scale both `X` and `y`
scaled_train_envs, scaled_test_envs = scale_environments(train_environments, test_environments)

print("Checking for NaNs in train/test environments...")
for i, (x, y) in enumerate(scaled_train_envs + scaled_test_envs):
    if torch.isnan(x).any() or torch.isnan(y).any():
        print(f"NaNs found in environment {i}!")

# Train IRM with scaled training environments
irm_model = IRM(scaled_train_envs, args)

# Train IRM model
#irm_model = IRM(environments, args)

# Get the final learned parameters
final_solution = irm_model.solution()
print("Final Learned Solution:", final_solution.detach().numpy())


##########################
### EVALUATE IRM MODEL ###
##########################
x_test, y_test = scaled_test_envs[0]
y_pred_scaled = x_test @ irm_model.solution()

# Compute and Print Test MSE (directly on scaled data)
mse = ((y_pred_scaled - y_test) ** 2).mean().item()
print("Test MSE (on scaled data):", mse)
