import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# Read in the data
df_xgb = pd.read_csv('/cluster/project/math/akmete/MSc/XGBoost/LOSO/results.csv')
df_lstm = pd.read_csv('/cluster/project/math/akmete/MSc/LSTM/LOSO/results.csv')
df_irm = pd.read_csv('/cluster/project/math/akmete/MSc/IRM/LOSO/results_global_scaling/results.csv')
df_lr = pd.read_csv('/cluster/project/math/akmete/MSc/IRM/LOSO/results_global_scaling/results.csv')

# Standardize column names for the common metrics
df_xgb.rename(columns={'r2_score': 'r2'}, inplace=True)
df_irm.rename(columns={'site_id': 'site'}, inplace=True)
df_lr.rename(columns={'site_left_out': 'site'}, inplace=True)
# df_lstm already has the correct column names for our purposes

# Define the list of common metrics
common_metrics = ['mse', 'rmse', 'r2', 'relative_error', 'mae']

# Dictionary mapping model names to their corresponding DataFrames
models = {
    'XGB': df_xgb,
    'LSTM': df_lstm,
    'IRM': df_irm,
    'LR': df_lr
}

# Create folder for saving plots if it doesn't exist
output_folder = 'plots'
os.makedirs(output_folder, exist_ok=True)

# Loop over each metric and create a boxplot comparing all models with moderate filtering of extreme outliers
for metric in common_metrics:
    plt.figure(figsize=(8, 6))
    
    plot_data = []
    for model_name, df in models.items():
        if metric in df.columns:
            values = df[metric]
            # Filter out the bottom and top 5% of values
            lower_bound = values.quantile(0.05)
            upper_bound = values.quantile(0.98)
            filtered_values = values[(values <= upper_bound)]
            
            temp = pd.DataFrame({
                'Model': model_name,
                'Value': filtered_values
            })
            plot_data.append(temp)
        else:
            print(f"Warning: {metric} not found in {model_name} dataframe.")
    
    if not plot_data:
        continue

    combined_df = pd.concat(plot_data, ignore_index=True)
    
    sns.boxplot(x='Model', y='Value', data=combined_df)
    plt.title(f'Comparison of {metric.upper()} across Models')
    plt.xlabel('Model')
    plt.ylabel(metric.upper())
    
    # Save the plot instead of displaying it
    plot_path = os.path.join(output_folder, f'{metric}_boxplot.png')
    plt.savefig(plot_path)
    plt.close()


'''
# Rename in df_irm 'site_id' into 'site_left_out'
df_irm = df_irm.rename(columns={'site_id': 'site_left_out', 'mse': 'mse_scaled', 'rmse': 'rmse_scaled'})

# Add suffix
df_xgb = df_xgb.add_suffix("_xgb")
df_lstm = df_lstm.add_suffix("_lstm")
df_irm = df_irm.add_suffix("_irm")

# Remove suffix from 'site' column
df_xgb = df_xgb.rename(columns={"site_left_out_xgb": "site"})
df_lstm = df_lstm.rename(columns={"site_left_out_lstm": "site"})
df_irm = df_irm.rename(columns={"site_left_out_irm": "site"})

# Merge dataframes on 'site'
df_merged = pd.merge(df_xgb, df_lstm, on="site", how="outer")
df_merged = pd.merge(df_merged, df_irm, on="site", how="outer")

# List of RMSE columns to process
rmse_columns = ["rmse_scaled_xgb", "rmse_scaled_lstm", "rmse_scaled_irm"]

'''
'''
# Function to remove outliers using IQR method
def remove_outliers(df, columns):
    for col in columns:
        Q1 = df[col].quantile(0.25)  # First quartile (25th percentile)
        Q3 = df[col].quantile(0.75)  # Third quartile (75th percentile)
        IQR = Q3 - Q1  # Interquartile range
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
    return df

# Remove outliers
df_filtered = remove_outliers(df_merged, rmse_columns)
'''
'''
# Convert to long format for plotting
df_melted = pd.melt(
    df_merged, 
    id_vars=["site"], 
    value_vars=rmse_columns,
    var_name="Model", 
    value_name="RMSE"
)

# Rename models for clarity
df_melted["Model"] = df_melted["Model"].replace({
    "rmse_scaled_xgb": "XGBoost",
    "rmse_scaled_lstm": "LSTM",
    "rmse_scaled_irm": "IRM"
})

# Box plot (without outliers)
plt.figure(figsize=(8, 6))
sns.boxplot(x="Model", y="RMSE", data=df_melted)
plt.xlabel("Model")
plt.ylabel("RMSE")
#plt.title("LOSO RMSE Distribution")
plt.savefig("XGB_LSTM_IRM_RMSEdistribution.png", dpi=300)

# Histogram with KDE (without outliers)
plt.figure(figsize=(8, 6))
sns.histplot(
    data=df_melted,
    x="RMSE",
    hue="Model",
    kde=True,
    alpha=0.5
)
plt.xlim(0, 0.05)
plt.xticks(np.arange(0, 0.05, 0.01))
plt.ticklabel_format(style='plain', axis='x')
plt.xlabel("RMSE")
plt.ylabel("Count of Sites")
#plt.title("LOSO Histogram of RMSE")
plt.tight_layout()
plt.savefig("histogram_RMSE.png", dpi=300)




# Another plot (lines to see whether increasing or decreasing from model to model)
# Randomly select 20 unique sites
selected_sites = np.random.choice(df_melted["site"].unique(), size=15, replace=False)

# Filter dataframe to only include selected sites
df_selected = df_melted[df_melted["site"].isin(selected_sites)]

# Define x-axis positions for each model
models = ["XGBoost", "LSTM", "IRM"]  # Corrected labels
x_positions = [1, 2, 3]  # XGB at x=1, LSTM at x=2, IRM at x=3

plt.figure(figsize=(12, 6))

# Iterate through selected sites and plot each site's values
for i, site in enumerate(selected_sites):  # Use the correct variable
    site_data = df_selected[df_selected["site"] == site]

    # Extract RMSE values correctly based on Model name
    values = [
        site_data.loc[site_data["Model"] == "XGBoost", "RMSE"].values,
        site_data.loc[site_data["Model"] == "LSTM", "RMSE"].values,
        site_data.loc[site_data["Model"] == "IRM", "RMSE"].values
    ]
    
    # Convert to list of floats (handling empty arrays)
    values = [v[0] if len(v) > 0 else np.nan for v in values]

    # Skip sites with NaNs
    if any(np.isnan(values)):
        continue

    # Plot the dots
    plt.scatter(x_positions, values, label=site if i < 1 else "", s=60)

    # Connect the dots with a line
    plt.plot(x_positions, values, linestyle='-', marker='o', alpha=0.7)

# Customize the plot
plt.xticks(x_positions, models)  # Set x-axis labels
plt.xlabel("Model")
plt.ylabel("RMSE")
#plt.title("RMSE LOSO Across Models")
plt.grid(True, linestyle="--", alpha=0.5)

# Show the plot
plt.tight_layout()
plt.savefig('test.png', dpi=300)'''