#!/bin/bash

# SLURM SUBMIT SCRIPT
#SBATCH --nodes=1
#SBATCH -p V100
#SBATCH --gres=gpu:1
##SBATCH --nodelist=gpunode51
#SBATCH --exclude=gpunode54
##SBATCH --ntasks-per-node=1
#SBATCH --time=30:00:00
#SBATCH --cpus-per-task=8
#SBATCH --hint=nomultithread
#SBATCH --qos=qos_gpu-t3
#SBATCH --output=./slurmlogs/log-finetune-time6h-%j.out
#SBATCH --error=./slurmlogs/log-finetune-time6h-%j.err

# Schedule execution of many runs
# Run from root folder with: bash scripts/schedule.sh

# srun python src/find_lr_and_train.py trainer=gpu trainer.devices=1 trainer.accelerator=gpu # trainer.max_epochs=1 # logger=tensorboard # debug=profiler 
# srun python src/find_lr.py trainer=gpu +trainer.accumulate_grad_batches=2 model=fourcastnet_finetune datamodule.split=finetune ckpt_path=/public/home/wangwuxing01/research/weatherbench/FourCastNet-PL/logs/train/runs/2022-11-29_12-35-05/checkpoints/last.ckpt
srun python src/finetune.py trainer=gpu trainer.max_epochs=25 +trainer.accumulate_grad_batches=2 model=fourcastnet_finetune datamodule.split=finetune ckpt_path=/public/home/wangwuxing01/research/weatherbench/FourCastNet-PL/logs/train/runs/2022-11-30_11-20-37/checkpoints/epoch_034.ckpt