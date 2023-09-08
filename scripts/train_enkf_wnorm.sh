#!/bin/bash

# SLURM SUBMIT SCRIPT
#SBATCH --nodes=1
#SBATCH -p GPU
#SBATCH --gres=gpu:1
#SBATCH --time=30:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=10G
#SBATCH --output=./slurmlogs/log-train-enkf-N40-partial1.0-wnorm-%j.out
#SBATCH --error=./slurmlogs/log-train-enkf-N40-partial1.0-wnorm-%j.err

# Schedule execution of many runs
# Run from root folder with: bash scripts/schedule.sh

#python src/data_factory/generate_dataset_L96.py --da_method='EnKF' --da_N=20 --da_Inf=1.02 --da_rot=True  --obs_partial=1.0 --years=4

python src/train.py model=tinynet data.obs_num=5 data.normalize=True
python src/train.py model=tinyresnet data.obs_num=5 data.normalize=True
python src/train.py model=resnet_v1 data.obs_num=5 data.normalize=True
python src/train.py model=resnet_v2 data.obs_num=5 data.normalize=True
python src/train.py model=seresnet_v1 data.obs_num=5 data.normalize=True
python src/train.py model=seresnet_v2 data.obs_num=5 data.normalize=True