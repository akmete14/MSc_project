# Import libraries
import pandas as pd
import matplotlib.pyplot as plt

# Load data
df_metrics = pd.read_csv('/cluster/project/math/akmete/MSc/XGBoost/LOSO/results.csv')
df_targets = pd.read_csv('/cluster/project/math/akmete/MSc/preprocessing/df_balanced_groups_onevegindex.csv')


# Sort metrics dataframe by RMSE in descrending order, then take worst 5 and best 5 sites
df_metrics_sorted = df_metrics.sort_values(by='rmse', ascending=False)

worst_n = 5
best_n = 5
worst_sites = df_metrics_sorted['site'].unique()[:worst_n]
best_sites = df_metrics_sorted['site'].unique()[-best_n:]

# Combine both into one list
sites_of_interest = list(best_sites) + list(worst_sites)

# Make boxplots
data_for_boxplot = []
colors = []

for site in sites_of_interest:

    site_data = df_targets.loc[df_targets['site_id'] == site, 'GPP'].dropna()
    data_for_boxplot.append(site_data)

    if site in worst_sites:
        colors.append('red')
    elif site in best_sites:
        colors.append('green')
    else:
        colors.append('gray')

fig, ax = plt.subplots(figsize=(8, 6))
box = ax.boxplot(data_for_boxplot,
                 vert=False,
                 patch_artist=True,
                 labels=sites_of_interest)

# 5) Set labels and title
ax.set_xlabel('GPP')
ax.set_ylabel('Sites')

plt.tight_layout()
plt.savefig('worst_5_best_5_gpp_distribution.png', dpi=300)