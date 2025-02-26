#!/bin/bash
#SBATCH --job-name=LOSO_LSTM
#SBATCH --output=slurm_%A_%a.out
#SBATCH --error=slurm_%A_%a.err
#SBATCH --array=0-289
#SBATCH --time=120:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=35840  # for example

# If you want a GPU, you might also include:
# #SBATCH --gres=gpu:1

module load stack/.2024-04-silent
module load stack/2024-04

source /cluster/project/math/akmete/Setup/.venv/bin/activate

echo "Starting array task $SLURM_ARRAY_TASK_ID"

# Run the Python script (now for LSTM), passing in the array index
python /cluster/project/math/akmete/MSc/LOSO/lstm_train_one_site.py $SLURM_ARRAY_TASK_ID
