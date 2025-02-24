#!/bin/bash

#SBATCH --job-name=loso_fold
#SBATCH --output=loso_fold_%A_%a.out
#SBATCH --error=loso_fold_%A_%a.err
#SBATCH --time=60:00:00
#SBATCH --mem-per-cpu=17000
#SBATCH --cpus-per-task=2
#SBATCH --array=0-4

module load stack/.2024-04-silent
module load stack/2024-04

module spider gcc/8.5.0

source /cluster/project/math/akmete/Setup/.venv/bin/activate

python /cluster/project/math/akmete/MSc/Stabilized_Regression/LOSO/5_sites_onevegindex/loso.py --fold ${SLURM_ARRAY_TASK_ID}
