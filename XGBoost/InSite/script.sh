#!/bin/bash

#SBATCH --job-name=in_site_xgb
#SBATCH --output=in_site_xgb_%A_%a.out
#SBATCH --error=in_site_xgb_%A_%a.err
#SBATCH --array=0-289    # e.g., if you have 50 sites, use 0-49
#SBATCH --time=10:00:00
#SBATCH --mem-per-cpu=20000

module load stack/.2024-04-silent
module load stack/2024-04
module spider gcc/8.5.0

source /cluster/project/math/akmete/Setup/.venv/bin/activate

# Run the in_site_xgboost.py script.
python /cluster/project/math/akmete/MSc/XGBoost/InSite/insite.py

