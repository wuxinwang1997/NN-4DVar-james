#!/bin/bash

# SLURM SUBMIT SCRIPT
#SBATCH --nodes=1
#SBATCH -p GPU
#SBATCH --gres=gpu:1
#SBATCH --time=30:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=10G
#SBATCH --output=./slurmlogs/log-train-enkf-N40-4dvarnet-partial1.0-20230524-%j.out
#SBATCH --error=./slurmlogs/log-train-enkf-N40-4dvarnet-partial1.0-20230524-%j.err

# Schedule execution of many runs
# Run from root folder with: bash scripts/schedule.sh

#python src/data_factory/generate_dataset_L96.py --da_method='EnKF' --da_N=20 --da_Inf=1.02 --da_rot=True  --obs_partial=1.0 --years=4

# python src/train.py model=fdvarnet_ae data=L96_4dvarnet_ienks_N40_partial1.0 data.obs_num=5 data.normalize=False test=False data.pred_len=1 data.da_method='iEnKS'
# python src/train.py model=fdvarnet_L96 data=L96_4dvarnet_ienks_N40_partial1.0 data.obs_num=5 data.normalize=False test=False data.pred_len=1 data.da_method='iEnKS'

# python src/train.py model=tinynet data=L96_ienks_N40_partial1.0 data.normalize=True data.pred_len=0 data.obs_num=5 data.da_method='iEnKS'
# python src/train.py model=tinyresnet data=L96_ienks_N40_partial1.0 data.normalize=True data.pred_len=0 data.obs_num=5 data.da_method='iEnKS'
# python src/train.py model=resnet_v1 data=L96_ienks_N40_partial1.0 data.normalize=True data.pred_len=0 data.obs_num=5 data.da_method='iEnKS'
# python src/train.py model=resnet_v2 data=L96_ienks_N40_partial1.0 data.normalize=True data.pred_len=0 data.obs_num=5 data.da_method='iEnKS'
python src/train.py model=seresnet_v1 data=L96_ienks_N40_partial1.0 data.normalize=True data.pred_len=0 data.obs_num=5 data.da_method='iEnKS'
python src/train.py model=seresnet_v2 data=L96_ienks_N40_partial1.0 data.normalize=True data.pred_len=0 data.obs_num=5 data.da_method='iEnKS' 

# python src/train.py model=tinynet data=L96_ienks_N40_partial1.0 data.normalize=True data.pred_len=1 data.obs_num=5 data.da_method='iEnKS'
# python src/train.py model=tinyresnet data=L96_ienks_N40_partial1.0 data.normalize=True data.pred_len=1 data.obs_num=5 data.da_method='iEnKS'
# python src/train.py model=resnet_v1 data=L96_ienks_N40_partial1.0 data.normalize=True data.pred_len=1 data.obs_num=5 data.da_method='iEnKS'
# python src/train.py model=resnet_v2 data=L96_ienks_N40_partial1.0 data.normalize=True data.pred_len=1 data.obs_num=5 data.da_method='iEnKS'
python src/train.py model=seresnet_v1 data=L96_ienks_N40_partial1.0 data.normalize=True data.pred_len=1 data.obs_num=5 data.da_method='iEnKS'
python src/train.py model=seresnet_v2 data=L96_ienks_N40_partial1.0 data.normalize=True data.pred_len=1 data.obs_num=5 data.da_method='iEnKS'

# python src/train.py model=tinynet data=L96_ienks_N40_partial1.0 data.normalize=True data.pred_len=2 data.obs_num=5 data.da_method='iEnKS'
# python src/train.py model=tinyresnet data=L96_ienks_N40_partial1.0 data.normalize=True data.pred_len=2 data.obs_num=5 data.da_method='iEnKS'
# python src/train.py model=resnet_v1 data=L96_ienks_N40_partial1.0 data.normalize=True data.pred_len=2 data.obs_num=5 data.da_method='iEnKS'
# python src/train.py model=resnet_v2 data=L96_ienks_N40_partial1.0 data.normalize=True data.pred_len=2 data.obs_num=5 data.da_method='iEnKS'
python src/train.py model=seresnet_v1 data=L96_ienks_N40_partial1.0 data.normalize=True data.pred_len=2 data.obs_num=5 data.da_method='iEnKS'
python src/train.py model=seresnet_v2 data=L96_ienks_N40_partial1.0 data.normalize=True data.pred_len=2 data.obs_num=5 data.da_method='iEnKS'

# python src/train.py model=tinynet data=L96_ienks_N40_partial1.0 data.normalize=True data.pred_len=3 data.obs_num=5 data.da_method='iEnKS'
# python src/train.py model=tinyresnet data=L96_ienks_N40_partial1.0 data.normalize=True data.pred_len=3 data.obs_num=5 data.da_method='iEnKS'
# python src/train.py model=resnet_v1 data=L96_ienks_N40_partial1.0 data.normalize=True data.pred_len=3 data.obs_num=5 data.da_method='iEnKS'
# python src/train.py model=resnet_v2 data=L96_ienks_N40_partial1.0 data.normalize=True data.pred_len=3 data.obs_num=5 data.da_method='iEnKS'
python src/train.py model=seresnet_v1 data=L96_ienks_N40_partial1.0 data.normalize=True data.pred_len=3 data.obs_num=5 data.da_method='iEnKS'
python src/train.py model=seresnet_v2 data=L96_ienks_N40_partial1.0 data.normalize=True data.pred_len=3 data.obs_num=5 data.da_method='iEnKS'

# python src/train.py model=tinynet data=L96_ienks_N40_partial1.0 data.normalize=True data.pred_len=4 data.obs_num=5 data.da_method='iEnKS'
# python src/train.py model=tinyresnet data=L96_ienks_N40_partial1.0 data.normalize=True data.pred_len=4 data.obs_num=5 data.da_method='iEnKS'
# python src/train.py model=resnet_v1 data=L96_ienks_N40_partial1.0 data.normalize=True data.pred_len=4 data.obs_num=5 data.da_method='iEnKS'
# python src/train.py model=resnet_v2 data=L96_ienks_N40_partial1.0 data.normalize=True data.pred_len=4 data.obs_num=5 data.da_method='iEnKS'
python src/train.py model=seresnet_v1 data=L96_ienks_N40_partial1.0 data.normalize=True data.pred_len=4 data.obs_num=5 data.da_method='iEnKS'
python src/train.py model=seresnet_v2 data=L96_ienks_N40_partial1.0 data.normalize=True data.pred_len=4 data.obs_num=5 data.da_method='iEnKS'

# python src/train.py model=tinynet data=L96_ienks_N40_partial1.0 data.normalize=True data.pred_len=5 data.obs_num=5 data.da_method='iEnKS'
# python src/train.py model=tinyresnet data=L96_ienks_N40_partial1.0 data.normalize=True data.pred_len=5 data.obs_num=5 data.da_method='iEnKS'
# python src/train.py model=resnet_v1 data=L96_ienks_N40_partial1.0 data.normalize=True data.pred_len=5 data.obs_num=5 data.da_method='iEnKS'
# python src/train.py model=resnet_v2 data=L96_ienks_N40_partial1.0 data.normalize=True data.pred_len=5 data.obs_num=5 data.da_method='iEnKS'
python src/train.py model=seresnet_v1 data=L96_ienks_N40_partial1.0 data.normalize=True data.pred_len=5 data.obs_num=5 data.da_method='iEnKS'
python src/train.py model=seresnet_v2 data=L96_ienks_N40_partial1.0 data.normalize=True data.pred_len=5 data.obs_num=5 data.da_method='iEnKS'

# python src/train.py model=tinynet data=L96_ienks_N40_partial1.0 data.normalize=True data.pred_len=6 data.obs_num=5 data.da_method='iEnKS'
# python src/train.py model=tinyresnet data=L96_ienks_N40_partial1.0 data.normalize=True data.pred_len=6 data.obs_num=5 data.da_method='iEnKS'
# python src/train.py model=resnet_v1 data=L96_ienks_N40_partial1.0 data.normalize=True data.pred_len=6 data.obs_num=5 data.da_method='iEnKS'
# python src/train.py model=resnet_v2 data=L96_ienks_N40_partial1.0 data.normalize=True data.pred_len=6 data.obs_num=5 data.da_method='iEnKS'
# python src/train.py model=seresnet_v1 data=L96_ienks_N40_partial1.0 data.normalize=True data.pred_len=6 data.obs_num=5 data.da_method='iEnKS'
# python src/train.py model=seresnet_v2 data=L96_ienks_N40_partial1.0 data.normalize=True data.pred_len=6 data.obs_num=5 data.da_method='iEnKS'

# python src/train.py model=tinynet data=L96_ienks_N40_partial1.0 data.normalize=True data.pred_len=7 data.obs_num=5 data.da_method='iEnKS'
# python src/train.py model=tinyresnet data=L96_ienks_N40_partial1.0 data.normalize=True data.pred_len=7 data.obs_num=5 data.da_method='iEnKS'
# python src/train.py model=resnet_v1 data=L96_ienks_N40_partial1.0 data.normalize=True data.pred_len=7 data.obs_num=5 data.da_method='iEnKS'
# python src/train.py model=resnet_v2 data=L96_ienks_N40_partial1.0 data.normalize=True data.pred_len=7 data.obs_num=5 data.da_method='iEnKS'
# python src/train.py model=seresnet_v1 data=L96_ienks_N40_partial1.0 data.normalize=True data.pred_len=7 data.obs_num=5 data.da_method='iEnKS'
# python src/train.py model=seresnet_v2 data=L96_ienks_N40_partial1.0 data.normalize=True data.pred_len=7 data.obs_num=5 data.da_method='iEnKS'

# python src/train.py model=tinynet data=L96_ienks_N40_partial1.0 data.normalize=True data.pred_len=8 data.obs_num=5 data.da_method='iEnKS'
# python src/train.py model=tinyresnet data=L96_ienks_N40_partial1.0 data.normalize=True data.pred_len=8 data.obs_num=5 data.da_method='iEnKS'
# python src/train.py model=resnet_v1 data=L96_ienks_N40_partial1.0 data.normalize=True data.pred_len=8 data.obs_num=5 data.da_method='iEnKS'
# python src/train.py model=resnet_v2 data=L96_ienks_N40_partial1.0 data.normalize=True data.pred_len=8 data.obs_num=5 data.da_method='iEnKS'
# python src/train.py model=seresnet_v1 data=L96_ienks_N40_partial1.0 data.normalize=True data.pred_len=8 data.obs_num=5 data.da_method='iEnKS'
# python src/train.py model=seresnet_v2 data=L96_ienks_N40_partial1.0 data.normalize=True data.pred_len=8 data.obs_num=5 data.da_method='iEnKS'