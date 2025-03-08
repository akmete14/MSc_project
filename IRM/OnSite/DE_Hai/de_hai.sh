#!/bin/bash

#SBATCH --job-name=insite_DEHai           # Job name
#SBATCH --output=logs/insite_DEHai_%j.out    # Standard output file (use %j for job ID)
#SBATCH --error=logs/insite_DEHai_%j.err     # Standard error file
#SBATCH --time=02:00:00                      # Maximum runtime (adjust as needed)
#SBATCH --mem-per-cpu=30000                             # Memory allocation (adjust as needed)
#SBATCH --cpus-per-task=1                    # Number of CPUs per task

# Load required modules
module load stack/2024-06 python_cuda/3.11.6

# Run the Python script that processes only the site 'DE-Hai'
python /cluster/project/math/akmete/MSc/IRM/OnSite/de_hai.py
