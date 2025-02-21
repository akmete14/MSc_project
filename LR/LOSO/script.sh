#!/bin/bash

#SBATCH --job-name=linreg_site
#SBATCH --output=linreg_site_%A_%a.out
#SBATCH --error=linreg_site_%A_%a.err
#SBATCH --array=0-289
#SBATCH --time=08:00:00
#SBATCH --mem-per-cpu=20000

module load stack/.2024-04-silent
module load stack/2024-04
module spider gcc/8.5.0

source /cluster/project/math/akmete/Setup/.venv/bin/activate


# Run the script (assuming it is named my_script.py)
python /cluster/project/math/akmete/MSc/LR/LOSO/loso.py
