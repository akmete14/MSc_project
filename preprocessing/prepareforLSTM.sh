#!/bin/bash

#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=01:00:00
#SBATCH --mem-per-cpu=35840
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=akmete@student.ethz.ch

module load stack/.2024-04-silent
module load stack/2024-04

module spider gcc/8.5.0

source /cluster/project/math/akmete/Setup/.venv/bin/activate

python /cluster/project/math/akmete/MSc/preprocessing/prepareforLSTM.py

