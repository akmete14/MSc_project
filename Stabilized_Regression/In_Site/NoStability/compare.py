import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend
import matplotlib.pyplot as plt

# Assume 'df' is your DataFrame with columns:
df = pd.read_csv('/cluster/project/math/akmete/MSc/Stabilized_Regression/In_Site/NoStability/results.csv')
# site, ensemble_mse_scaled, full_mse_scaled, G_hat_count, O_hat_count, ensemble_rmse, full_rmse

# Melt the dataframe so that RMSE values for each model are in one column:
df_melted = df.melt(
    id_vars=['site'],
    value_vars=['ensemble_rmse', 'full_rmse'],
    var_name='Model',
    value_name='RMSE'
)

# Optionally, rename the model labels for clarity:
df_melted['Model'] = df_melted['Model'].replace({
    'ensemble_rmse': 'Stabilized Regression',
    'full_rmse': 'Full Model'
})

# Create the histogram with KDE
plt.figure(figsize=(8, 6))
sns.histplot(
    data=df_melted,
    x="RMSE",
    hue="Model",
    kde=True,
    alpha=0.5
)
plt.xlim(0, 0.05)
plt.xticks(np.arange(0, 0.1, 0.01))
plt.ticklabel_format(style='plain', axis='x')
plt.xlabel("RMSE")
plt.ylabel("Count of Sites")
plt.tight_layout()
plt.savefig("histogram_RMSE.png", dpi=300)
#plt.show()

# Assume your DataFrame is named df and has columns 'ensemble_rmse' and 'full_rmse'
plt.figure(figsize=(8, 6))
plt.scatter(df['full_rmse'], df['ensemble_rmse'], color='blue', alpha=0.7)
plt.xlabel('Full Model RMSE')
plt.ylabel('Stabilized Regression RMSE')
#plt.title('Scatter Plot of Stabilized vs Full Model RMSE per Site')

# Determine the limits for the plot to ensure the diagonal covers the range
min_rmse = min(df['full_rmse'].min(), df['ensemble_rmse'].min())
max_rmse = max(df['full_rmse'].max(), df['ensemble_rmse'].max())
plt.plot([min_rmse, max_rmse], [min_rmse, max_rmse], 'r--', label='x = y')

plt.legend()
plt.tight_layout()
plt.savefig("scatter_rmse.png", dpi=300)
#plt.show()