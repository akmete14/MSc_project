#!/bin/bash

#SBATCH --job-name=insite_regression
#SBATCH --output=logs/insite_%A_%a.out
#SBATCH --error=logs/insite_%A_%a.err
#SBATCH --array=0-289
#SBATCH --time=48:00:00
#SBATCH --mem-per-cpu=30000

# Load modules and activate venv
module load stack/.2024-04-silent
module load stack/2024-04
module spider gcc/8.5.0

source /cluster/project/math/akmete/Setup/.venv/bin/activate

# Run the Python script with the current array task ID as site_index
python /cluster/project/math/akmete/MSc/Stabilized_Regression/In_Site/NoStability/XGBoost/insite_screened.py --site_index $SLURM_ARRAY_TASK_ID

