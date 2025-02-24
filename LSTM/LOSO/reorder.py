import pandas as pd

# Load the CSV file
df = pd.read_csv("/cluster/project/math/akmete/MSc/LSTM/LOSO/results_LOSO.csv")

# Sort by 'site' column
df_sorted = df.sort_values(by="site")

# Save the sorted DataFrame back to CSV
df_sorted.to_csv("results.csv", index=False)

# Display the sorted DataFrame
print(df_sorted)
