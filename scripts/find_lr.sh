#!/bin/bash

# SLURM SUBMIT SCRIPT
#SBATCH --nodes=1
#SBATCH -p V100
#SBATCH --gres=gpu:1
##SBATCH --exclude=gpunode54
##SBATCH --ntasks-per-node=1
#SBATCH --time=30:00:00
#SBATCH --cpus-per-task=8
#SBATCH --hint=nomultithread
#SBATCH --qos=qos_gpu-t3
#SBATCH --output=./slurmlogs/log-findlr-pretrain-time6h-%j.out
#SBATCH --error=./slurmlogs/log-findlr-pretrain-time6h-%j.err

# Schedule execution of many runs
# Run from root folder with: bash scripts/schedule.sh

srun python src/find_lr.py trainer=gpu trainer.devices=1 trainer.accelerator=gpu # trainer.max_epochs=1 # logger=tensorboard # debug=profiler 
