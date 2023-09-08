#!/bin/bash

# SLURM SUBMIT SCRIPT
#SBATCH --nodes=1
#SBATCH -p GPU
#SBATCH --gres=gpu:1
#SBATCH --time=30:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=10G
#SBATCH --output=./slurmlogs/log-train-enkf-N40-partial1.0-wonorm-predlen3-%j.out
#SBATCH --error=./slurmlogs/log-train-enkf-N40-partial1.0-wonorm-predlen3-%j.err

# Schedule execution of many runs
# Run from root folder with: bash scripts/schedule.sh

#python src/data_factory/generate_dataset_L96.py --da_method='EnKF' --da_N=20 --da_Inf=1.02 --da_rot=True  --obs_partial=1.0 --years=4

python src/train.py model=tinynet_v1 data.normalize=False data.pred_len=3
python src/train.py model=tinynet_v2 data.normalize=False data.pred_len=3
python src/train.py model=tinyresnet_v1 data.normalize=False data.pred_len=3
python src/train.py model=tinyresnet_v2 data.normalize=False data.pred_len=3
python src/train.py model=resnet_v1 data.normalize=False data.pred_len=3
python src/train.py model=resnet_v2 data.normalize=False data.pred_len=3
python src/train.py model=resnet_v3 data.normalize=False data.pred_len=3
python src/train.py model=resnet_v4 data.normalize=False data.pred_len=3
python src/train.py model=resnet_v5 data.normalize=False data.pred_len=3
python src/train.py model=resnet_v6 data.normalize=False data.pred_len=3
python src/train.py model=resnet_v7 data.normalize=False data.pred_len=3
python src/train.py model=resnet_v8 data.normalize=False data.pred_len=3
python src/train.py model=resnet_v9 data.normalize=False data.pred_len=3
python src/train.py model=resnet_v10 data.normalize=False data.pred_len=3
python src/train.py model=resnet_v11 data.normalize=False data.pred_len=3
python src/train.py model=resnet_v12 data.normalize=False data.pred_len=3
python src/train.py model=seresnet_v1 data.normalize=False data.pred_len=3
python src/train.py model=seresnet_v2 data.normalize=False data.pred_len=3
python src/train.py model=seresnet_v3 data.normalize=False data.pred_len=3
python src/train.py model=seresnet_v4 data.normalize=False data.pred_len=3
python src/train.py model=seresnet_v5 data.normalize=False data.pred_len=3
python src/train.py model=seresnet_v6 data.normalize=False data.pred_len=3
python src/train.py model=seresnet_v7 data.normalize=False data.pred_len=3
python src/train.py model=seresnet_v8 data.normalize=False data.pred_len=3
python src/train.py model=seresnet_v9 data.normalize=False data.pred_len=3
python src/train.py model=seresnet_v10 data.normalize=False data.pred_len=3
python src/train.py model=seresnet_v11 data.normalize=False data.pred_len=3
python src/train.py model=seresnet_v12 data.normalize=False data.pred_len=3
