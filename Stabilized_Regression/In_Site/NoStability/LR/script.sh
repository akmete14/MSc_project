#!/bin/bash

#SBATCH --job-name=insite_stabreg
#SBATCH --output=logs/insite_%A_%a.out
#SBATCH --error=logs/insite_%A_%a.err
#SBATCH --array=0-289
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --mem-per-cpu=20000

module load stack/.2024-04-silent
module load stack/2024-04
module spider gcc/8.5.0

source /cluster/project/math/akmete/Setup/.venv/bin/activate

python /cluster/project/math/akmete/MSc/Stabilized_Regression/In_Site/NoStability/LR/insite.py --site_index ${SLURM_ARRAY_TASK_ID}

