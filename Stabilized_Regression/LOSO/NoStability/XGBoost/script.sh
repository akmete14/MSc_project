#!/bin/bash

#SBATCH --job-name=xgb_stabilized_regression
#SBATCH --output=logs/xgb_%A_%a.out
#SBATCH --error=logs/xgb_%A_%a.err
#SBATCH --array=0-49
#SBATCH --time=72:00:00
#SBATCH --mem-per-cpu=35840
#SBATCH --cpus-per-task=3

# Load necessary modules or activate venv
module load stack/.2024-04-silent
module load stack/2024-04
module spider gcc/8.5.0

source /cluster/project/math/akmete/Setup/.venv/bin/activate

# Run the Python script
python /cluster/project/math/akmete/MSc/Stabilized_Regression/LOSO/NoStability/XGBoost/loso_screened.py ${SLURM_ARRAY_TASK_ID}
