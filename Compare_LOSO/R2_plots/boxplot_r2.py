import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load datasets
df_xgb = pd.read_csv('/cluster/project/math/akmete/MSc/XGBoost/LOSO/results.csv')
df_lstm = pd.read_csv('/cluster/project/math/akmete/MSc/LSTM/LOSO/results.csv')
df_irm = pd.read_csv('/cluster/project/math/akmete/MSc/IRM/LOSO/results_global_scaling/results.csv')
df_lr = pd.read_csv('/cluster/project/math/akmete/MSc/LR/LOSO/results_2.csv')

# Standardize column names
df_xgb.rename(columns={'r2_score': 'r2'}, inplace=True)
df_irm.rename(columns={'site_id': 'site'}, inplace=True)
df_lr.rename(columns={'site_left_out': 'site'}, inplace=True)

# Add model identifiers
df_xgb['Model'] = 'XGBoost'
df_lstm['Model'] = 'LSTM'
df_irm['Model'] = 'IRM'
df_lr['Model'] = 'LR'

# Select relevant columns
df_lr = df_lr[['Model', 'r2']]
df_xgb = df_xgb[['Model', 'r2']]
df_lstm = df_lstm[['Model', 'r2']]
df_irm = df_irm[['Model', 'r2']]

# Function to remove extreme outliers using the 99% quantile
def remove_upper_outliers(df, column='r2'):
    Q01 = df[column].quantile(0.15)  # 85th percentile
    return df[df[column] >= Q01]  # Keep values below this threshold


# Apply outlier removal
df_lr_filtered = remove_upper_outliers(df_lr)
df_irm_filtered = remove_upper_outliers(df_irm)
df_xgb_filtered = remove_upper_outliers(df_xgb)
df_lstm_filtered = remove_upper_outliers(df_lstm)

# Combine all models into a single DataFrame
df_combined = pd.concat([df_lr_filtered, df_xgb_filtered, df_lstm_filtered, df_irm_filtered], ignore_index=True)

# Remove negative R² values (if applicable)
#df_combined = df_combined[df_combined['r2'] >= 0]

# ===============================
# 📌 Box Plot of R² Scores Across Models
# ===============================
plt.figure(figsize=(8, 6))

sns.boxplot(
    data=df_combined, 
    x="Model", 
    y="r2", 
    #palette="Set2",  # Nice color scheme
    width=0.6,  # Adjust box width
    showfliers=True  # Show outliers (can be set to False if you want to hide them)
)

# Formatting
plt.xlabel("Model")
plt.ylabel("R² Score")
#plt.title("Comparison of R² Scores Across Models")

# Set dynamic y-axis limits based on data range
plt.ylim(df_combined['r2'].min() - 0.05, df_combined['r2'].max() + 0.15)

# Display the plot
plt.tight_layout()
plt.savefig("boxplot_R2.png", dpi=300)
