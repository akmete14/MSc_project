#!/bin/bash
#SBATCH --job-name=onsite_prediction
#SBATCH --output=onsite_%A_%a.out   # %A: Job array ID, %a: Array index
#SBATCH --error=onsite_%A_%a.err
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1       
#SBATCH --mem-per-cpu=35840
#SBATCH --array=0-289

# Load required modules
module load stack/2024-06 python_cuda/3.11.6

# Run the Python script with the array index as the site index argument
python /cluster/project/math/akmete/MSc/IRM/OnSite/irm.py ${SLURM_ARRAY_TASK_ID}

