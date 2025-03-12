import pandas as pd
import torch
from torch.autograd import grad
import numpy as np
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

print("Modules and torch imported")

############################
### MAKE DATAFRAME READY ###
############################

# Read the CSV file and clean it
df = pd.read_csv('/cluster/project/math/akmete/MSc/preprocessing/df_balanced_groups_onevegindex.csv')
df = df.dropna(axis=1, how='all')  # Drop columns where all values are NaN
df = df.fillna(0)
df = df.drop(columns=['Unnamed: 0', 'cluster'])

# Cast float64 columns to float32 for efficiency
for col in tqdm(df.select_dtypes(include=['float64']).columns, desc="Casting columns"):
    df[col] = df[col].astype('float32')

print(df.columns)
print(df.head())
print(f"Total rows: {len(df)}")

# Identify feature columns and target
feature_columns = [col for col in df.columns if col not in ['GPP', 'site_id']]
target_column = "GPP"

# Prepare environments: each environment corresponds to one site
environments = []
site_order = []  # store site names in order
for site, group in df.groupby('site_id'):
    X = torch.tensor(group[feature_columns].values, dtype=torch.float32)
    y = torch.tensor(group['GPP'].values, dtype=torch.float32).view(-1, 1)
    environments.append((X, y))
    site_order.append(site)
    print(f"Site {site}: X shape = {X.shape}, y shape = {y.shape}")

# Define global scaling: using training environments to compute min/max
def scale_environments_global(train_environments, test_environments):
    all_train_x = torch.cat([env[0] for env in train_environments], dim=0)
    all_train_y = torch.cat([env[1] for env in train_environments], dim=0)
    
    # Compute global min and max for features
    min_x, _ = torch.min(all_train_x, dim=0)
    max_x, _ = torch.max(all_train_x, dim=0)
    diff_x = max_x - min_x
    diff_x[diff_x == 0] = 1  # avoid division by zero
    
    # Compute global min and max for target
    min_y, _ = torch.min(all_train_y, dim=0)
    max_y, _ = torch.max(all_train_y, dim=0)
    epsilon = 1e-8  # to avoid division by zero

    # Scale training environments
    scaled_train_envs = []
    for (x, y) in train_environments:
        scaled_x = (x - min_x) / diff_x
        scaled_y = (y - min_y) / (max_y - min_y + epsilon)
        scaled_train_envs.append((scaled_x, scaled_y))
    
    # Scale test environments using same parameters
    scaled_test_envs = []
    for (x, y) in test_environments:
        scaled_x = (x - min_x) / diff_x
        scaled_y = (y - min_y) / (max_y - min_y + epsilon)
        scaled_test_envs.append((scaled_x, scaled_y))
    
    return scaled_train_envs, scaled_test_envs

################
### IRM CLASS ###
################

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
        # Initialize phi as an identity matrix and w as ones
        self.phi = torch.nn.Parameter(torch.eye(dim_x, dim_x))
        self.w = torch.ones(dim_x, 1, requires_grad=True)

        opt = torch.optim.Adam([self.phi], lr=args["lr"])
        loss_fn = torch.nn.MSELoss()

        best_loss = float('inf')
        patience_counter = 0
        patience = args.get("early_stopping_patience", 100)
        min_delta = args.get("early_stopping_min_delta", 1e-4)

        for iteration in range(args["n_iterations"]):
            penalty = 0
            error = 0
            for x_e, y_e in environments:
                error_e = loss_fn(x_e @ self.phi @ self.w, y_e)
                penalty += grad(error_e, self.w, create_graph=True)[0].pow(2).mean()
                error += error_e

            total_loss = reg * error + (1 - reg) * penalty

            opt.zero_grad()
            total_loss.backward()
            opt.step()

            current_loss = total_loss.item()

            if current_loss < best_loss - min_delta:
                best_loss = current_loss
                patience_counter = 0
            else:
                patience_counter += 1

            if args["verbose"] and iteration % 1000 == 0:
                print(f"Iteration {iteration:05d} | Reg: {reg:.5f} | Error: {error:.5f} | "
                      f"Penalty: {penalty:.5f} | Total Loss: {current_loss:.5f}")

            if patience_counter >= patience:
                if args["verbose"]:
                    print(f"Early stopping at iteration {iteration} with best total loss {best_loss:.5f}")
                break

    def solution(self):
        return (self.phi @ self.w).view(-1, 1)

################
### LOEO-CV  ###
################

# Training parameters for IRM
args_irm = {
    "lr": 0.01,
    "n_iterations": 1000,
    "verbose": False,  # Set to True for detailed logging
    "early_stopping_patience": 100,
    "early_stopping_min_delta": 1e-4
}

site_ids = np.array(site_order)  # Array of site names

def run_fold(fold_index):
    # Leave out the environment (site) specified by fold_index
    train_environments = [env for j, env in enumerate(environments) if j != fold_index]
    test_environment = [environments[fold_index]]

    # Scale environments using global parameters from training data
    scaled_train_envs, scaled_test_envs = scale_environments_global(train_environments, test_environment)
    
    # Train IRM model on the scaled training environments
    irm_model = IRM(scaled_train_envs, args_irm)
    
    # Evaluate on the left-out test environment
    x_test, y_test = scaled_test_envs[0]
    y_pred_scaled = x_test @ irm_model.solution()
    
    # Save the predictions and actual values
    actual = y_test.detach().numpy().flatten()
    predicted = y_pred_scaled.detach().numpy().flatten()
    predictions_df = pd.DataFrame({'actual': actual, 'predicted': predicted})
    predictions_df.to_csv("de_hai_irm_predictions.csv", index=False)
    print("Predictions and actual values saved to de_hai_irm_predictions.csv")
    
    # Create a joint scatter plot with marginal histograms
    g = sns.jointplot(x='actual', y='predicted', data=predictions_df, kind="scatter", s=1)
    plt.plot([0, 1], [0, 1], color="red")  # Reference line y=x
    plt.xlabel("True GPP")
    plt.ylabel("Predicted GPP")
    plt.savefig('de_hai_irm_IRM.png')
    plt.show()
    
if __name__ == '__main__':
    # Instead of using command-line arguments, find the index for the 'DE-Hai' site
    try:
        fold_index = list(site_ids).index("DE-Hai")
    except ValueError:
        print("DE-Hai site not found in the dataset.")
        exit(1)
    
    run_fold(fold_index)

