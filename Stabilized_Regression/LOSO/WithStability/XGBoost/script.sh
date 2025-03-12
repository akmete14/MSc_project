#!/bin/bash

#SBATCH --job-name=xgb_loso_nostability
#SBATCH --output=logs/xgb_loso_nostability_%A_%a.out
#SBATCH --error=logs/xgb_loso_nostability_%A_%a.err
#SBATCH --array=0-49
#SBATCH --time=120:00:00
#SBATCH --mem-per-cpu=35840
#SBATCH --cpus-per-task=2

# Load the required modules and activate venv
module load stack/.2024-04-silent
module load stack/2024-04
module spider gcc/8.5.0

source /cluster/project/math/akmete/Setup/.venv/bin/activate

# Execute the Python script with the SLURM_ARRAY_TASK_ID as an argument.
python /cluster/project/math/akmete/MSc/Stabilized_Regression/LOSO/WithStability/XGBoost/loso_screened.py $SLURM_ARRAY_TASK_ID
