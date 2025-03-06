#!/bin/bash

#SBATCH --job-name=logo_xgb_stab
#SBATCH --output=logs/logo_xgb_%A_%a.out
#SBATCH --error=logs/logo_xgb_%A_%a.err
#SBATCH --array=0-9
#SBATCH --time=120:00:00
#SBATCH --cpus-per-task=3
#SBATCH --mem-per-cpu=40000

# Load modules and activate venv
module load stack/.2024-04-silent
module load stack/2024-04
module spider gcc/8.5.0

source /cluster/project/math/akmete/Setup/.venv/bin/activate

# Run the LOGO stabilized regression script with test_cluster set to the array task ID.
python /cluster/project/math/akmete/MSc/Stabilized_Regression/LOGO/WithStability/XGBoost/logo_screened.py --test_cluster $SLURM_ARRAY_TASK_ID

