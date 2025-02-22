#!/bin/bash

#SBATCH --job-name=logo_xgb_weighted
#SBATCH --output=logo_xgb_weighted_%A_%a.out
#SBATCH --error=logo_xgb_weighted_%A_%a.err
#SBATCH --array=0-9
#SBATCH --time=16:00:00
#SBATCH --mem-per-cpu=35840

module load stack/.2024-04-silent
module load stack/2024-04
module spider gcc/8.5.0

source /cluster/project/math/akmete/Setup/.venv/bin/activate

python /cluster/project/math/akmete/MSc/XGBoost/LOGO/balanced_grouping_weighted_training/logo.py

