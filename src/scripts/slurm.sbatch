#!/bin/bash
#
#SBATCH --job-name=train-masked-irl
#SBATCH --account=clear
#SBATCH --partition=shared-v100,shared-a6000,shared-l40s,shared-rtx4090,shared-rtx6000ada,shared-a100,shared-h100,shared-h200
#SBATCH --qos=shared-if-available
#SBATCH --time=7:00:00 # 12 hours
#SBATCH --output=/{PATH_TO_MASKED_IRL}/logs/%x_%j.out
#SBATCH --error=/{PATH_TO_MASKED_IRL}/logs/%x_%j.err
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB


# Your job commands go here
echo "Training Masked IRL"
source /data/clear/robot-simulation/Installations/anaconda3/etc/profile.d/conda.sh
eval "$(conda shell.bash hook)"
conda activate maskedirl
cd /{PATH_TO_MASKED_IRL}/src


set -euo pipefail

# Only create the symlink if it does not already exist
if [ ! -e ~/\.wandb_api_key ]; then
	ln -s /{PATH_TO_MASKED_IRL}/.wandb_api_key ~/ 
fi

# echo arguments
# echo "$@"

# Forward all args to the main script
bash ./scripts/train.sh "$@"