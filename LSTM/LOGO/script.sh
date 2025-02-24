#!/bin/bash

#SBATCH --job-name=logo_lstm
#SBATCH --output=logo_lstm_%A_%a.out
#SBATCH --error=logo_lstm_%A_%a.err
#SBATCH --array=0-9
#SBATCH --time=96:00:00
#SBATCH --mem-per-cpu=28000

module load stack/.2024-04-silent
module load stack/2024-04
module spider gcc/8.5.0

source /cluster/project/math/akmete/Setup/.venv/bin/activate

python /cluster/project/math/akmete/MSc/LSTM/LOGO/logo.py

