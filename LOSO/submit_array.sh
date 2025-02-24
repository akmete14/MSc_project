#!/bin/bash
#SBATCH --job-name=LOSO_XGB
#SBATCH --output=slurm_%A_%a.out
#SBATCH --error=slurm_%A_%a.err
#SBATCH --array=0-289      # Launch tasks for each index
#SBATCH --time=24:00:00    # e.g. 24 hours
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=20480

module load stack/.2024-04-silent
module load stack/2024-04

module spider gcc/8.5.0

source /cluster/project/math/akmete/Setup/.venv/bin/activate

echo "Starting array task $SLURM_ARRAY_TASK_ID"

# Run the Python script, passing in the array index
python /cluster/project/math/akmete/MSc/LOSO/xgb_train_one_site.py  $SLURM_ARRAY_TASK_ID

