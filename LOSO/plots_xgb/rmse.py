import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

df = pd.read_csv('/cluster/project/math/akmete/MSc/LOSO/xgb.csv')

plt.figure(figsize=(6, 4))
sns.boxplot(y='rmse_scaled', data=df)
plt.ylabel('RMSE')
plt.title('Distribution of RMSE across Sites')
plt.savefig('rmse_boxplot_xgb')


import numpy as np

# Create the histogram plot
plt.figure(figsize=(6, 4))
sns.histplot(data=df, x="rmse_scaled", kde=True)

# Set x-axis limits
plt.xlim(0, 0.05)

# Explicitly set x-axis tick locations
plt.xticks(np.arange(0, 0.051, 0.01))  # Adjusts the scale with steps of 0.01

# Ensure numerical formatting without scientific notation
plt.ticklabel_format(style='plain', axis='x')

# Label the x-axis and y-axis
plt.xlabel("RMSE")
plt.ylabel("Count")

# Set the title
# plt.title("Histogram of RMSE")

# Save the plot
plt.savefig('histogram_RMSE')
