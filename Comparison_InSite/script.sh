#!/bin/bash

#SBATCH --job-name=insite_DEHai           # Job name
#SBATCH --output=logs/insite_DEHai_%j.out    # Standard output file (use %j for job ID)
#SBATCH --error=logs/insite_DEHai_%j.err     # Standard error file
#SBATCH --time=00:10:00                      # Maximum runtime (adjust as needed)
#SBATCH --mem-per-cpu=30000                             # Memory allocation (adjust as needed)

# Load required modules and activate venv
module load stack/.2024-04-silent
module load stack/2024-04
module spider gcc/8.5.0

source /cluster/project/math/akmete/Setup/.venv/bin/activate

python /cluster/project/math/akmete/MSc/Comparison_InSite/test_plots.py
