import pandas as pd
import glob

all_files = glob.glob("results_*.csv")
dfs = [pd.read_csv(f) for f in all_files]
final_results = pd.concat(dfs, ignore_index=True)
final_results.to_csv("all_sites_results_xgb.csv", index=False)
