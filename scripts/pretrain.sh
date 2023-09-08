#!/bin/bash

# SLURM SUBMIT SCRIPT
#SBATCH --nodes=1
#SBATCH -p V100
#SBATCH --gres=gpu:1
##SBATCH --ntasks-per-node=1
#SBATCH --time=30:00:00
#SBATCH --cpus-per-task=8
#SBATCH --hint=nomultithread
#SBATCH --qos=qos_gpu-t3
#SBATCH --output=./slurmlogs/log-pretrain-time6h-%j.out
#SBATCH --error=./slurmlogs/log-pretrain-time6h-%j.err

# Schedule execution of many runs
# Run from root folder with: bash scripts/schedule.sh

srun python src/train.py --config-name=train_pred datamodule.lead_time=3
#srun -N 1 -p V100 --gres=gpu:1 python src/pretrain_continue.py