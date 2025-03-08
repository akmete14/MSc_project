import pandas as pd
import torch
from torch.autograd import grad
import numpy as np
import os
from tqdm import tqdm

print("Modules and torch imported.")

# --- Define the scaling function ---
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

    def minmax_scale_y(y):
        return (y - min_y) / (max_y - min_y + 1e-8)

    scaled_train_envs = [(minmax_scale_x(x), minmax_scale_y(y)) for x, y in train_environments]
    scaled_test_envs = [(minmax_scale_x(x), minmax_scale_y(y)) for x, y in test_environments]

    return scaled_train_envs, scaled_test_envs

# --- Define the IRM class ---
class IRM:
    def __init__(self, environments, args):
        best_reg = 0
        best_err = 1e6
        x_val = environments[-1][0]  
        y_val = environments[-1][1]

        for reg in [0, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]:
            self.train(environments[:-1], args, reg=reg)
            err = (x_val @ self.solution() - y_val).pow(2).mean().item()

            print(f"IRM (reg={reg:.3f}) has validation error: {err:.3f}")

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
        loss_fn = torch.nn.MSELoss()

        for iteration in range(args["n_iterations"]):
            penalty = 0
            error = 0
            for x_e, y_e in environments:
                error_e = loss_fn(x_e @ self.phi @ self.w, y_e)
                penalty += grad(error_e, self.w, create_graph=True)[0].pow(2).mean()
                error += error_e

            opt.zero_grad()
            (reg * error + (1 - reg) * penalty).backward()
            opt.step()

            if args["verbose"] and iteration % 1000 == 0:
                print(f"Iteration {iteration:05d} | Reg: {reg:.5f} | Error: {error:.5f} | Penalty: {penalty:.5f}")

    def solution(self):
        return (self.phi @ self.w).view(-1, 1)

######################
######## MAIN ########
######################

def main():
    # Hardcode the target site to 'DE-Hai'
    chosen_site = "DE-Hai"
    print("Processing site:", chosen_site)
    
    # Load full dataset and preprocess
    df = pd.read_csv('/cluster/project/math/akmete/MSc/preprocessing/df_balanced_groups_onevegindex.csv')
    df = df.dropna(axis=1, how='all')
    df = df.fillna(0)
    df = df.drop(columns=['Unnamed: 0', 'cluster'])
    for col in tqdm(df.select_dtypes(include=['float64']).columns, desc="Casting columns"):
        df[col] = df[col].astype('float32')

    # Define features and target
    feature_columns = [col for col in df.columns if col not in ['GPP', 'site_id']]
    target_column = "GPP"

    # Filter data for the chosen site
    site_data = df[df['site_id'] == chosen_site].copy()

    # Perform an 80/20 chronological split
    n = len(site_data)
    split_idx = int(0.8 * n)
    train_df = site_data.iloc[:split_idx]
    test_df = site_data.iloc[split_idx:]
    print(f"Total samples: {n} | Training: {len(train_df)} | Testing: {len(test_df)}")

    # Convert to PyTorch tensors
    X_train = torch.tensor(train_df[feature_columns].values, dtype=torch.float32)
    y_train = torch.tensor(train_df[target_column].values, dtype=torch.float32).view(-1, 1)
    X_test = torch.tensor(test_df[feature_columns].values, dtype=torch.float32)
    y_test = torch.tensor(test_df[target_column].values, dtype=torch.float32).view(-1, 1)

    train_env = (X_train, y_train)
    test_env = (X_test, y_test)

    # Scale the environments using the training data
    scaled_train_envs, scaled_test_envs = scale_environments([train_env], [test_env])

    # Set training parameters
    args_params = {"lr": 0.01, "n_iterations": 1000, "verbose": False}

    # Train the IRM model on the combined environments (train + test)
    irm_model = IRM(scaled_train_envs + scaled_test_envs, args_params)
    final_solution = irm_model.solution()
    print("Final learned solution:", final_solution.detach().numpy())

    # Evaluate on the test environment
    x_test_scaled, y_test_scaled = scaled_test_envs[0]
    y_pred_scaled = x_test_scaled @ irm_model.solution()

    mse = ((y_pred_scaled - y_test_scaled) ** 2).mean().item()
    rmse = np.sqrt(mse)
    ss_res = ((y_test_scaled - y_pred_scaled) ** 2).sum()
    ss_tot = ((y_test_scaled - y_test_scaled.mean()) ** 2).sum()
    r2 = (1 - ss_res / ss_tot).item()
    mae = torch.abs(y_test_scaled - y_pred_scaled).mean().item()
    relative_error = (torch.abs(y_test_scaled - y_pred_scaled) / torch.abs(y_test_scaled)).mean().item()

    print("Test MSE:", mse, "RMSE:", rmse, "R2:", r2, "Relative Error:", relative_error, "MAE:", mae)

    # Save overall results to a CSV file
    result = pd.DataFrame({
        'site_id': [chosen_site],
        'mse': [mse],
        'rmse': [rmse],
        'r2': [r2],
        'relative_error': [relative_error],
        'mae': [mae]
    })
    os.makedirs("results_onsite", exist_ok=True)
    output_filename = os.path.join("results_onsite", f"onsite_{chosen_site}.csv")
    result.to_csv(output_filename, index=False)
    print("Overall results saved to", output_filename)

    # --- Save per-sample predictions ---
    # Convert predictions and actual values to NumPy arrays
    y_pred_np = y_pred_scaled.detach().cpu().numpy().flatten()
    y_test_np = y_test_scaled.detach().cpu().numpy().flatten()

    predictions_df = pd.DataFrame({
        'site': chosen_site,
        'method': 'IRM',
        'actual': y_test_np,
        'predicted': y_pred_np
    })
    predictions_filename = os.path.join("results_onsite", f"predictions_{chosen_site}.csv")
    predictions_df.to_csv(predictions_filename, index=False)
    print("Predictions saved to", predictions_filename)

if __name__ == '__main__':
    main()
