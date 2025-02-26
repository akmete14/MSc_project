#!/bin/bash

#SBATCH --job-name=loso_array
#SBATCH --output=loso_array_%A_%a.out
#SBATCH --error=loso_array_%A_%a.err
#SBATCH --time=120:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=35840
#SBATCH --array=0-289

# Load modules and activate venv
module load stack/.2024-04-silent
module load stack/2024-04
module spider gcc/8.5.0

source /cluster/project/math/akmete/Setup/.venv/bin/activate

# Run the modified LOSO script.
python /cluster/project/math/akmete/MSc/Stabilized_Regression/LOSO/NoStability/LR/loso_lasso.py

