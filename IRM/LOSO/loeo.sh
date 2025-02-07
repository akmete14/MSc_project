#!/bin/bash

#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --time=15:00:00
#SBATCH --mem-per-cpu=35840
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=akmete@student.ethz.ch

#module load stack/.2024-04-silent
#module load stack/2024-04

#module spider gcc/8.5.0

#source /cluster/project/math/akmete/Setup/.venv/bin/activate

# Load required modules
module load stack/2024-06 python_cuda/3.11.6

python /cluster/project/math/akmete/MSc/IRM/LOSO/loeo.py

