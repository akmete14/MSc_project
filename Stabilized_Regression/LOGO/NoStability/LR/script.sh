#!/bin/bash

#SBATCH --job-name=logo_stab_reg
#SBATCH --output=logo_stab_reg_%A_%a.out
#SBATCH --error=logo_stab_reg_%A_%a.err
#SBATCH --time=80:00:00
#SBATCH --mem-per-cpu=30000
#SBATCH --array=0-9

# Load modules and activate venv
module load stack/.2024-04-silent
module load stack/2024-04
module spider gcc/8.5.0

source /cluster/project/math/akmete/Setup/.venv/bin/activate

# Run the LOGO stabilized regression script with the test_cluster set to the array task ID.
python /cluster/project/math/akmete/MSc/Stabilized_Regression/LOGO/NoStability/LR/logo_lasso.py --test_cluster ${SLURM_ARRAY_TASK_ID}

