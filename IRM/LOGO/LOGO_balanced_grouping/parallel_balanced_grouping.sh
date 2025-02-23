#!/bin/bash
#SBATCH --job-name=LOGO
#SBATCH --output=loeo_%A_%a.out   # %A is the job array ID, %a is the array index
#SBATCH --error=loeo_%A_%a.err
#SBATCH --time=120:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=35840
#SBATCH --array=0-9

# Load required modules
module load stack/2024-06 python_cuda/3.11.6

# Run the Python script, passing the SLURM_ARRAY_TASK_ID as the fold index
python /cluster/project/math/akmete/MSc/IRM/LOGO/balanced_grouping.py ${SLURM_ARRAY_TASK_ID}

