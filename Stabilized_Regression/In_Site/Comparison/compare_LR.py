import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats

# File paths (update these paths as needed)
csv_no_stability = '/cluster/project/math/akmete/MSc/Stabilized_Regression/In_Site/NoStability/LR/results_lasso_modified.csv'
csv_stability = '/cluster/project/math/akmete/MSc/Stabilized_Regression/In_Site/WithStability/LR/results_lasso_modified.csv'

# Load the CSV files into DataFrames
df_no_stability = pd.read_csv(csv_no_stability)
df_stability = pd.read_csv(csv_stability)
print(df_no_stability['site'].head())
print(df_stability['site'].head())


# Columns in df_no_stability: site,ensemble_mse_scaled,full_mse_scaled,ensemble_rmse_scaled,full_rmse_scaled,ensemble_r2,full_r2,ensemble_relative_error,full_relative_error,ensemble_mae,full_mae,O_hat_count
# Columns in df_stability: site,ensemble_mse_scaled,full_mse_scaled,ensemble_rmse_scaled,full_rmse_scaled,ensemble_r2,full_r2,ensemble_relative_error,full_relative_error,ensemble_mae,full_mae,G_hat_count,O_hat_count,O_hat

# File paths (update these paths if needed)
csv_no_stability = '/cluster/project/math/akmete/MSc/Stabilized_Regression/In_Site/NoStability/LR/results_lasso_modified.csv'
csv_stability = '/cluster/project/math/akmete/MSc/Stabilized_Regression/In_Site/WithStability/LR/results_lasso_modified.csv'

# Load CSV files
df_no_stability = pd.read_csv(csv_no_stability)
df_stability = pd.read_csv(csv_stability)

# Merge both dataframes on "site"
df_comparison = df_no_stability.merge(df_stability, on="site", suffixes=("_nostab", "_withstab"))

# Compute RMSE differences (Ensemble)
df_comparison["rmse_diff_ensemble"] = df_comparison["ensemble_rmse_scaled_nostab"] - df_comparison["ensemble_rmse_scaled_withstab"]

# Summary statistics
print(df_comparison[["site", "rmse_diff_ensemble"]].describe())

### 📊 1. Bar Plot: RMSE Differences Per Site
plt.figure(figsize=(10, 6))
sns.barplot(data=df_comparison, x="site", y="rmse_diff_ensemble", palette="viridis")
plt.axhline(0, color='red', linestyle='dashed')  # Reference line at 0
plt.xticks(rotation=90)
plt.xlabel("Site")
plt.ylabel("RMSE Difference (No Stability - With Stability)")
plt.title("Change in Ensemble RMSE Per Site")
plt.savefig('nostaminusstab.png')

### 📊 2. Boxplot for RMSE Comparison
plt.figure(figsize=(8, 6))
df_melt = df_comparison.melt(id_vars="site", value_vars=["ensemble_rmse_scaled_nostab", "ensemble_rmse_scaled_withstab"],
                             var_name="Model", value_name="Ensemble RMSE")
sns.boxplot(data=df_melt, x="Model", y="Ensemble RMSE")
plt.title("Ensemble RMSE Distribution: No Stability vs. With Stability")
plt.savefig('rmseboxplot.png')

### 📊 3. Scatter Plot: RMSE No Stability vs. With Stability
plt.figure(figsize=(8, 8))
sns.scatterplot(data=df_comparison, x="ensemble_rmse_scaled_nostab", y="ensemble_rmse_scaled_withstab")
plt.plot([df_comparison["ensemble_rmse_scaled_nostab"].min(), df_comparison["ensemble_rmse_scaled_nostab"].max()],
         [df_comparison["ensemble_rmse_scaled_nostab"].min(), df_comparison["ensemble_rmse_scaled_nostab"].max()],
         linestyle="dashed", color="red")  # y=x reference line
plt.xlabel("Ensemble RMSE No Stability")
plt.ylabel("Ensemble RMSE With Stability")
plt.title("Comparison of Ensemble RMSE per Site")
plt.savefig('persitecomparison.png')

### 🧪 4. Statistical Tests
# Paired t-test (if normally distributed)
t_stat, p_ttest = stats.ttest_rel(df_comparison["ensemble_rmse_scaled_nostab"], df_comparison["ensemble_rmse_scaled_withstab"])
print(f"Paired t-test: t={t_stat:.4f}, p={p_ttest:.4f}")

# Wilcoxon signed-rank test (non-parametric alternative)
wil_stat, p_wilcoxon = stats.wilcoxon(df_comparison["ensemble_rmse_scaled_nostab"], df_comparison["ensemble_rmse_scaled_withstab"])
print(f"Wilcoxon test: W={wil_stat:.4f}, p={p_wilcoxon:.4f}")

# Interpretation
if p_ttest < 0.05 or p_wilcoxon < 0.05:
    print("There is a significant difference between No Stability and With Stability RMSEs.")
else:
    print("No significant difference found between the two models.")


'''
# Drop columns that are available in the stability DataFrame but not in the no_stability DataFrame
df_stability = df_stability.drop(columns=['G_hat_count', 'O_hat'])

# Merge the DataFrames on the 'site' column; add suffixes to distinguish between the two versions
merged_df = df_no_stability.merge(df_stability, on='site', suffixes=('_no_stab', '_stab'))

# Define the ensemble metric columns to compare
ensemble_metrics = [
    'ensemble_mse_scaled',
    'ensemble_rmse_scaled',
    'ensemble_r2',
    'ensemble_relative_error',
    'ensemble_mae'
]

# Create a folder to save the plots if it doesn't exist
plots_folder = "plots_LR"
os.makedirs(plots_folder, exist_ok=True)

# Loop through each metric, remove outliers, and create a scatter plot
for metric in ensemble_metrics:
    x_metric = f"{metric}_no_stab"
    y_metric = f"{metric}_stab"
    
    # Calculate the IQR for the no_stability column
    Q1_x = merged_df[x_metric].quantile(0.25)
    Q3_x = merged_df[x_metric].quantile(0.75)
    IQR_x = Q3_x - Q1_x
    lower_bound_x = Q1_x - 1.5 * IQR_x
    upper_bound_x = Q3_x + 1.5 * IQR_x
    
    # Calculate the IQR for the with_stability column
    Q1_y = merged_df[y_metric].quantile(0.25)
    Q3_y = merged_df[y_metric].quantile(0.75)
    IQR_y = Q3_y - Q1_y
    lower_bound_y = Q1_y - 1.5 * IQR_y
    upper_bound_y = Q3_y + 1.5 * IQR_y
    
    # Filter rows where both columns fall within the computed bounds
    filtered_df = merged_df[(merged_df[x_metric] >= lower_bound_x) & (merged_df[x_metric] <= upper_bound_x) &
                              (merged_df[y_metric] >= lower_bound_y) & (merged_df[y_metric] <= upper_bound_y)]
    
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=filtered_df, x=x_metric, y=y_metric)
    
    # Plot the line of equality (45° line)
    line_min = min(filtered_df[x_metric].min(), filtered_df[y_metric].min())
    line_max = max(filtered_df[x_metric].max(), filtered_df[y_metric].max())
    plt.plot([line_min, line_max], [line_min, line_max], ls="--", color="red")
    
    plt.title(f"Comparison of {metric.replace('_', ' ').title()} (Outliers Removed)")
    plt.xlabel("No Stability")
    plt.ylabel("With Stability")
    plt.grid(True)
    
    # Save the figure to the specified folder
    plot_filename = os.path.join(plots_folder, f"{metric}_comparison_no_outliers.png")
    plt.savefig(plot_filename, bbox_inches='tight')
    plt.close()
'''