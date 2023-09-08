#!/bin/bash

# SLURM SUBMIT SCRIPT
#SBATCH --nodes=1
#SBATCH -p normal
#SBATCH --time=30:00:00
#SBATCH --cpus-per-task=8
#SBATCH --hint=nomultithread
#SBATCH --qos=qos_gpu-t3
#SBATCH --output=./slurmlogs/tensorboard-%j.out
#SBATCH --error=./slurmlogs/tensorboard-%j.err

# Schedule execution of many runs
# Run from root folder with: bash scripts/schedule.sh

tensorboard --logdir=logs/train/runs/2023-03-31_14-30-16/tensorboard/ --port=6006 --bind_all
