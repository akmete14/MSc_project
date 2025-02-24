#!/bin/bash

#SBATCH --job-name=loso_lstm
#SBATCH --output=loso_lstm_%A_%a.out
#SBATCH --error=loso_lstm_%A_%a.err
#SBATCH --array=0-289
#SBATCH --time=96:00:00
#SBATCH --mem-per-cpu=35840

module load stack/.2024-04-silent
module load stack/2024-04
module spider gcc/8.5.0

source /cluster/project/math/akmete/Setup/.venv/bin/activate

python /cluster/project/math/akmete/MSc/LSTM/LOSO/loso.py

