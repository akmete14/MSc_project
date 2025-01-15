#!/bin/bash

#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=01:00:00
#SBATCH --mem-per-cpu=1024

module load stack/2024-04

module spider gcc/8.5.0

source ~/Setup/.venv/bin/activate

python ~/MSc/LOSO3sites.py
