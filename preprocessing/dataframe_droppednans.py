import pandas as pd

df = pd.read_csv('/cluster/project/math/akmete/MSc/preprocessing/df_balanced_groups.csv')
df = df.drop(columns=['Unnamed: 0'])
df = df.dropna(axis=1, how='all')  # Drop columns where all values are NaN
df = df.fillna(0)

print(df.head())
df.to_csv('df_balanced_groups_nonans.csv')
print("df_balanced_groups_nonans saved.")