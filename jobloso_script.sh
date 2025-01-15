#!/bin/bash

#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --mem-per-cpu=4096
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=akmete@student.ethz.ch

module load stack/2024-04

module spider gcc/8.5.0

source ~/Setup/.venv/bin/activate

python ~/MSc/backupLOSOxgboost.py
