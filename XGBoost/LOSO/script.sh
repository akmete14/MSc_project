#!/bin/bash

#SBATCH --job-name=loso_xgb
#SBATCH --output=loso_xgb_%A_%a.out
#SBATCH --error=loso_xgb_%A_%a.err
#SBATCH --array=0-289    # Replace <max_index> with number of sites minus one (e.g., 0-49)
#SBATCH --time=24:00:00
#SBATCH --mem-per-cpu=35840

module load stack/.2024-04-silent
module load stack/2024-04
module spider gcc/8.5.0

source /cluster/project/math/akmete/Setup/.venv/bin/activate

python /cluster/project/math/akmete/MSc/XGBoost/LOSO/loso.py

