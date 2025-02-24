#!/bin/bash

#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=120:00:00
#SBATCH --mem-per-cpu=61440
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=akmete@student.ethz.ch

module load stack/.2024-04-silent
module load stack/2024-04

module spider gcc/8.5.0

source /cluster/project/math/akmete/Setup/.venv/bin/activate

python /cluster/project/math/akmete/MSc/LSTM/LOGO_balanced_grouping.py
