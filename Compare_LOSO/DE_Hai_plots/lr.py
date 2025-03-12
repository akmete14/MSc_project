import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

# Load your CSV files
df = pd.read_csv('/cluster/project/math/akmete/MSc/LR/LOSO/DE_Hai/results_DE-Hai_linear.csv')



# Extract the actual and predicted values
y_test_standard = df["actual"]
y_pred = df["predicted"]

# Enable interactive mode
# plt.ion()

# Create the joint scatter plot
g = sns.jointplot(x=y_test_standard, y=y_pred, kind="scatter", s=1)

# Add the y=x reference line
plt.plot([0.35, 0.8], [0.35, 0.8], color="red")

# Set axis labels
plt.xlabel("True GPP")
plt.ylabel("Predicted GPP")

# Show the interactive plot
plt.savefig('de_hai_LR.png')
