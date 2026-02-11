#!/bin/bash
#
#SBATCH --job-name=train-masked-irl
#SBATCH --account=clear
#SBATCH --partition=clear-l40s
#SBATCH --qos=clear-main
#SBATCH --time=04:00:00 # 4 hours
#SBATCH --output=/data/clear/robot-simulation/pybullet-franka-sim/logs/%x_%j.out
#SBATCH --error=/data/clear/robot-simulation/pybullet-franka-sim/logs/%x_%j.err
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

ln -s /data/clear/robot-simulation/.wandb_api_key ~/

# Forward all args to the main script
bash ./scripts/train.sh "$@"