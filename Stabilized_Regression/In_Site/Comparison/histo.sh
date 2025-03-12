#!/bin/bash

#SBATCH --job-name=insite_comparison
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:10:00
#SBATCH --mem-per-cpu=35840

# Load required modules and activate venv
module load stack/.2024-04-silent
module load stack/2024-04
module spider gcc/8.5.0

source /cluster/project/math/akmete/Setup/.venv/bin/activate

# Run python script
python /cluster/project/math/akmete/MSc/Stabilized_Regression/In_Site/Comparison/histoLR.py
