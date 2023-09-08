#!/bin/bash

# SLURM SUBMIT SCRIPT
#SBATCH --nodes=1
#SBATCH -p normal
##SBATCH --gres=gpu:1
##SBATCH --nodelist=gpunode51
##SBATCH --time=30:00:00
#SBATCH --mem=10G
#SBATCH --cpus-per-task=1
#SBATCH --hint=nomultithread
#SBATCH --qos=qos_gpu-t3
#SBATCH --output=./slurmlogs/genrate_data_for_L96_with_100observations-%j.out
#SBATCH --error=./slurmlogs/genrate_data_for_L96_with_100observations-%j.err

# python src/data_factory/generate_dataset_L96.py --da_method='My4DVar' --da_Lag=1 --da_Bx=0.04 --obs_partial=1.0
python src/data_factory/generate_dataset_L96.py --da_method='EnKF' --da_N=20 --da_Inf=1.04 --da_rot=True --obs_partial=0.75
# python src/data_factory/generate_dataset_L96.py --da_method='iEnKS' --da_N=20 --da_Inf=1.02 --da_Lag=1 --da_rot=False --obs_partial=1.0
# python src/data_factory/generate_dataset_L96.py --da_method='LETKF' --da_N=20 --da_Inf=1.02 --da_rot=True --da_loc_rad=10 --obs_partial=1.0

# python src/evaluation/search_param_traditional_da.py --da_method='EnKF' --da_N=20 --da_Inf=1.0 --da_rot=True  --obs_partial=0.75
# python src/evaluation/search_param_traditional_da.py --da_method='EnKF' --da_N=20 --da_Inf=1.0 --da_rot=Flase --obs_partial=0.75
# python src/evaluation/search_param_traditional_da.py --da_method='EnKF' --da_N=20 --da_Inf=1.02 --da_rot=True --obs_partial=0.75
# python src/evaluation/search_param_traditional_da.py --da_method='EnKF' --da_N=20 --da_Inf=1.02 --da_rot=False --obs_partial=0.75
# python src/evaluation/search_param_traditional_da.py --da_method='EnKF' --da_N=20 --da_Inf=1.04 --da_rot=True --obs_partial=0.75
# python src/evaluation/search_param_traditional_da.py --da_method='EnKF' --da_N=20 --da_Inf=1.04 --da_rot=False --obs_partial=0.75
# python src/evaluation/search_param_traditional_da.py --da_method='EnKF' --da_N=20 --da_Inf=1.06 --da_rot=True --obs_partial=0.75
# python src/evaluation/search_param_traditional_da.py --da_method='EnKF' --da_N=20 --da_Inf=1.06 --da_rot=False --obs_partial=0.75
# python src/evaluation/search_param_traditional_da.py --da_method='EnKF' --da_N=20 --da_Inf=1.08 --da_rot=True --obs_partial=0.75
# python src/evaluation/search_param_traditional_da.py --da_method='EnKF' --da_N=20 --da_Inf=1.08 --da_rot=False --obs_partial=0.75
# python src/evaluation/search_param_traditional_da.py --da_method='EnKF' --da_N=20 --da_Inf=1.1 --da_rot=True --obs_partial=0.75
# python src/evaluation/search_param_traditional_da.py --da_method='EnKF' --da_N=20 --da_Inf=1.1 --da_rot=False --obs_partial=0.75
##
# python src/evaluation/search_param_traditional_da.py --da_method='iEnKS' --da_N=20 --da_Inf=1.0 --da_Lag=1 --da_rot=True --obs_partial=0.75
# python src/evaluation/search_param_traditional_da.py --da_method='iEnKS' --da_N=20 --da_Inf=1.0 --da_Lag=1 --da_rot=Flase --obs_partial=0.75
# python src/evaluation/search_param_traditional_da.py --da_method='iEnKS' --da_N=20 --da_Inf=1.02 --da_Lag=1 --da_rot=True --obs_partial=0.75
# python src/evaluation/search_param_traditional_da.py --da_method='iEnKS' --da_N=20 --da_Inf=1.02 --da_Lag=1 --da_rot=False --obs_partial=0.75
# python src/evaluation/search_param_traditional_da.py --da_method='iEnKS' --da_N=20 --da_Inf=1.04 --da_Lag=1 --da_rot=True --obs_partial=0.75
# python src/evaluation/search_param_traditional_da.py --da_method='iEnKS' --da_N=20 --da_Inf=1.04 --da_Lag=1 --da_rot=False --obs_partial=0.75
# python src/evaluation/search_param_traditional_da.py --da_method='iEnKS' --da_N=20 --da_Inf=1.06 --da_Lag=1 --da_rot=True --obs_partial=0.75
# python src/evaluation/search_param_traditional_da.py --da_method='iEnKS' --da_N=20 --da_Inf=1.06 --da_Lag=1 --da_rot=False --obs_partial=0.75
# python src/evaluation/search_param_traditional_da.py --da_method='iEnKS' --da_N=20 --da_Inf=1.08 --da_Lag=1 --da_rot=True --obs_partial=0.75
# python src/evaluation/search_param_traditional_da.py --da_method='iEnKS' --da_N=20 --da_Inf=1.08 --da_Lag=1 --da_rot=False --obs_partial=0.75
# python src/evaluation/search_param_traditional_da.py --da_method='iEnKS' --da_N=20 --da_Inf=1.1 --da_Lag=1 --da_rot=True --obs_partial=0.75
# python src/evaluation/search_param_traditional_da.py --da_method='iEnKS' --da_N=20 --da_Inf=1.1 --da_Lag=1 --da_rot=False --obs_partial=0.75
##
# python src/evaluation/search_param_traditional_da.py --da_method='LETKF' --da_N=20 --da_Inf=1.0 --da_rot=True --da_loc_rad=2 --obs_partial=0.75
# python src/evaluation/search_param_traditional_da.py --da_method='LETKF' --da_N=20 --da_Inf=1.0 --da_rot=False --da_loc_rad=2 --obs_partial=0.75
# python src/evaluation/search_param_traditional_da.py --da_method='LETKF' --da_N=20 --da_Inf=1.0 --da_rot=True --da_loc_rad=4 --obs_partial=0.75
# python src/evaluation/search_param_traditional_da.py --da_method='LETKF' --da_N=20 --da_Inf=1.0 --da_rot=False --da_loc_rad=4 --obs_partial=0.75
# python src/evaluation/search_param_traditional_da.py --da_method='LETKF' --da_N=20 --da_Inf=1.0 --da_rot=True --da_loc_rad=6 --obs_partial=0.75
# python src/evaluation/search_param_traditional_da.py --da_method='LETKF' --da_N=20 --da_Inf=1.0 --da_rot=False --da_loc_rad=6 --obs_partial=0.75
# python src/evaluation/search_param_traditional_da.py --da_method='LETKF' --da_N=20 --da_Inf=1.0 --da_rot=True --da_loc_rad=8 --obs_partial=0.75
# python src/evaluation/search_param_traditional_da.py --da_method='LETKF' --da_N=20 --da_Inf=1.0 --da_rot=False --da_loc_rad=8 --obs_partial=0.75
# python src/evaluation/search_param_traditional_da.py --da_method='LETKF' --da_N=20 --da_Inf=1.0 --da_rot=True --da_loc_rad=10 --obs_partial=0.75
# python src/evaluation/search_param_traditional_da.py --da_method='LETKF' --da_N=20 --da_Inf=1.0 --da_rot=False --da_loc_rad=10 --obs_partial=0.75
## python src/evaluation/search_param_traditional_da.py --da_method='LETKF' --da_N=20 --da_Inf=1.0 --da_rot=True --da_loc_rad=6 --obs_partial=0.75
## python src/evaluation/search_param_traditional_da.py --da_method='LETKF' --da_N=20 --da_Inf=1.0 --da_rot=False --da_loc_rad=6 --obs_partial=0.75
## python src/evaluation/search_param_traditional_da.py --da_method='LETKF' --da_N=20 --da_Inf=1.0 --da_rot=True --da_loc_rad=7 --obs_partial=0.75
## python src/evaluation/search_param_traditional_da.py --da_method='LETKF' --da_N=20 --da_Inf=1.0 --da_rot=False --da_loc_rad=7 --obs_partial=0.75
# python src/evaluation/search_param_traditional_da.py --da_method='LETKF' --da_N=20 --da_Inf=1.02 --da_rot=True --da_loc_rad=2 --obs_partial=0.75
# python src/evaluation/search_param_traditional_da.py --da_method='LETKF' --da_N=20 --da_Inf=1.02 --da_rot=False --da_loc_rad=2 --obs_partial=0.75
# python src/evaluation/search_param_traditional_da.py --da_method='LETKF' --da_N=20 --da_Inf=1.02 --da_rot=True --da_loc_rad=4 --obs_partial=0.75
# python src/evaluation/search_param_traditional_da.py --da_method='LETKF' --da_N=20 --da_Inf=1.02 --da_rot=False --da_loc_rad=4 --obs_partial=0.75
# python src/evaluation/search_param_traditional_da.py --da_method='LETKF' --da_N=20 --da_Inf=1.02 --da_rot=True --da_loc_rad=6 --obs_partial=0.75
# python src/evaluation/search_param_traditional_da.py --da_method='LETKF' --da_N=20 --da_Inf=1.02 --da_rot=False --da_loc_rad=6 --obs_partial=0.75
# python src/evaluation/search_param_traditional_da.py --da_method='LETKF' --da_N=20 --da_Inf=1.02 --da_rot=True --da_loc_rad=8 --obs_partial=0.75
# python src/evaluation/search_param_traditional_da.py --da_method='LETKF' --da_N=20 --da_Inf=1.02 --da_rot=False --da_loc_rad=8 --obs_partial=0.75
# python src/evaluation/search_param_traditional_da.py --da_method='LETKF' --da_N=20 --da_Inf=1.02 --da_rot=True --da_loc_rad=10 --obs_partial=0.75
# python src/evaluation/search_param_traditional_da.py --da_method='LETKF' --da_N=20 --da_Inf=1.02 --da_rot=False --da_loc_rad=10 --obs_partial=0.75
## python src/evaluation/search_param_traditional_da.py --da_method='LETKF' --da_N=20 --da_Inf=1.02 --da_rot=True --da_loc_rad=6 --obs_partial=0.75
## python src/evaluation/search_param_traditional_da.py --da_method='LETKF' --da_N=20 --da_Inf=1.02 --da_rot=False --da_loc_rad=6 --obs_partial=0.75
## python src/evaluation/search_param_traditional_da.py --da_method='LETKF' --da_N=20 --da_Inf=1.02 --da_rot=True --da_loc_rad=7 --obs_partial=0.75
## python src/evaluation/search_param_traditional_da.py --da_method='LETKF' --da_N=20 --da_Inf=1.02 --da_rot=False --da_loc_rad=7 --obs_partial=0.75
# python src/evaluation/search_param_traditional_da.py --da_method='LETKF' --da_N=20 --da_Inf=1.04 --da_rot=True --da_loc_rad=2 --obs_partial=0.75
# python src/evaluation/search_param_traditional_da.py --da_method='LETKF' --da_N=20 --da_Inf=1.04 --da_rot=False --da_loc_rad=2 --obs_partial=0.75
# python src/evaluation/search_param_traditional_da.py --da_method='LETKF' --da_N=20 --da_Inf=1.04 --da_rot=True --da_loc_rad=4 --obs_partial=0.75
# python src/evaluation/search_param_traditional_da.py --da_method='LETKF' --da_N=20 --da_Inf=1.04 --da_rot=False --da_loc_rad=4 --obs_partial=0.75
# python src/evaluation/search_param_traditional_da.py --da_method='LETKF' --da_N=20 --da_Inf=1.04 --da_rot=True --da_loc_rad=6 --obs_partial=0.75
# python src/evaluation/search_param_traditional_da.py --da_method='LETKF' --da_N=20 --da_Inf=1.04 --da_rot=False --da_loc_rad=6 --obs_partial=0.75
# python src/evaluation/search_param_traditional_da.py --da_method='LETKF' --da_N=20 --da_Inf=1.04 --da_rot=True --da_loc_rad=8 --obs_partial=0.75
# python src/evaluation/search_param_traditional_da.py --da_method='LETKF' --da_N=20 --da_Inf=1.04 --da_rot=False --da_loc_rad=8 --obs_partial=0.75
# python src/evaluation/search_param_traditional_da.py --da_method='LETKF' --da_N=20 --da_Inf=1.04 --da_rot=True --da_loc_rad=10 --obs_partial=0.75
# python src/evaluation/search_param_traditional_da.py --da_method='LETKF' --da_N=20 --da_Inf=1.04 --da_rot=False --da_loc_rad=10 --obs_partial=0.75
## python src/evaluation/search_param_traditional_da.py --da_method='LETKF' --da_N=20 --da_Inf=1.04 --da_rot=True --da_loc_rad=6 --obs_partial=0.75
## python src/evaluation/search_param_traditional_da.py --da_method='LETKF' --da_N=20 --da_Inf=1.04 --da_rot=False --da_loc_rad=6 --obs_partial=0.75
## python src/evaluation/search_param_traditional_da.py --da_method='LETKF' --da_N=20 --da_Inf=1.04 --da_rot=True --da_loc_rad=7 --obs_partial=0.75
## python src/evaluation/search_param_traditional_da.py --da_method='LETKF' --da_N=20 --da_Inf=1.04 --da_rot=False --da_loc_rad=7 --obs_partial=0.75
# python src/evaluation/search_param_traditional_da.py --da_method='LETKF' --da_N=20 --da_Inf=1.06 --da_rot=True --da_loc_rad=2 --obs_partial=0.75
# python src/evaluation/search_param_traditional_da.py --da_method='LETKF' --da_N=20 --da_Inf=1.06 --da_rot=False --da_loc_rad=2 --obs_partial=0.75
# python src/evaluation/search_param_traditional_da.py --da_method='LETKF' --da_N=20 --da_Inf=1.06 --da_rot=True --da_loc_rad=4 --obs_partial=0.75
# python src/evaluation/search_param_traditional_da.py --da_method='LETKF' --da_N=20 --da_Inf=1.06 --da_rot=False --da_loc_rad=4 --obs_partial=0.75
# python src/evaluation/search_param_traditional_da.py --da_method='LETKF' --da_N=20 --da_Inf=1.06 --da_rot=True --da_loc_rad=6 --obs_partial=0.75
# python src/evaluation/search_param_traditional_da.py --da_method='LETKF' --da_N=20 --da_Inf=1.06 --da_rot=Fasle --da_loc_rad=6 --obs_partial=0.75
# python src/evaluation/search_param_traditional_da.py --da_method='LETKF' --da_N=20 --da_Inf=1.06 --da_rot=True --da_loc_rad=8 --obs_partial=0.75
# python src/evaluation/search_param_traditional_da.py --da_method='LETKF' --da_N=20 --da_Inf=1.06 --da_rot=False --da_loc_rad=8 --obs_partial=0.75
# python src/evaluation/search_param_traditional_da.py --da_method='LETKF' --da_N=20 --da_Inf=1.06 --da_rot=True --da_loc_rad=10 --obs_partial=0.75
# python src/evaluation/search_param_traditional_da.py --da_method='LETKF' --da_N=20 --da_Inf=1.06 --da_rot=False --da_loc_rad=10 --obs_partial=0.75
## python src/evaluation/search_param_traditional_da.py --da_method='LETKF' --da_N=20 --da_Inf=1.06 --da_rot=True --da_loc_rad=6 --obs_partial=0.75
## python src/evaluation/search_param_traditional_da.py --da_method='LETKF' --da_N=20 --da_Inf=1.06 --da_rot=False --da_loc_rad=6 --obs_partial=0.75
## python src/evaluation/search_param_traditional_da.py --da_method='LETKF' --da_N=20 --da_Inf=1.06 --da_rot=True --da_loc_rad=7 --obs_partial=0.75
## python src/evaluation/search_param_traditional_da.py --da_method='LETKF' --da_N=20 --da_Inf=1.06 --da_rot=False --da_loc_rad=7 --obs_partial=0.75
# python src/evaluation/search_param_traditional_da.py --da_method='LETKF' --da_N=20 --da_Inf=1.08 --da_rot=True --da_loc_rad=2 --obs_partial=0.75
# python src/evaluation/search_param_traditional_da.py --da_method='LETKF' --da_N=20 --da_Inf=1.08 --da_rot=False --da_loc_rad=2 --obs_partial=0.75
# python src/evaluation/search_param_traditional_da.py --da_method='LETKF' --da_N=20 --da_Inf=1.08 --da_rot=True --da_loc_rad=4 --obs_partial=0.75
# python src/evaluation/search_param_traditional_da.py --da_method='LETKF' --da_N=20 --da_Inf=1.08 --da_rot=False --da_loc_rad=4 --obs_partial=0.75
# python src/evaluation/search_param_traditional_da.py --da_method='LETKF' --da_N=20 --da_Inf=1.08 --da_rot=True --da_loc_rad=6 --obs_partial=0.75
# python src/evaluation/search_param_traditional_da.py --da_method='LETKF' --da_N=20 --da_Inf=1.08 --da_rot=Fasle --da_loc_rad=6 --obs_partial=0.75
# python src/evaluation/search_param_traditional_da.py --da_method='LETKF' --da_N=20 --da_Inf=1.08 --da_rot=True --da_loc_rad=8 --obs_partial=0.75
# python src/evaluation/search_param_traditional_da.py --da_method='LETKF' --da_N=20 --da_Inf=1.08 --da_rot=False --da_loc_rad=8 --obs_partial=0.75
# python src/evaluation/search_param_traditional_da.py --da_method='LETKF' --da_N=20 --da_Inf=1.08 --da_rot=True --da_loc_rad=10 --obs_partial=0.75
# python src/evaluation/search_param_traditional_da.py --da_method='LETKF' --da_N=20 --da_Inf=1.08 --da_rot=False --da_loc_rad=10 --obs_partial=0.75
## python src/evaluation/search_param_traditional_da.py --da_method='LETKF' --da_N=20 --da_Inf=1.08 --da_rot=True --da_loc_rad=6 --obs_partial=0.75
## python src/evaluation/search_param_traditional_da.py --da_method='LETKF' --da_N=20 --da_Inf=1.08 --da_rot=False --da_loc_rad=6 --obs_partial=0.75
## python src/evaluation/search_param_traditional_da.py --da_method='LETKF' --da_N=20 --da_Inf=1.08 --da_rot=True --da_loc_rad=7 --obs_partial=0.75
## python src/evaluation/search_param_traditional_da.py --da_method='LETKF' --da_N=20 --da_Inf=1.08 --da_rot=False --da_loc_rad=7 --obs_partial=0.75
# python src/evaluation/search_param_traditional_da.py --da_method='LETKF' --da_N=20 --da_Inf=1.1 --da_rot=True --da_loc_rad=2 --obs_partial=0.75
# python src/evaluation/search_param_traditional_da.py --da_method='LETKF' --da_N=20 --da_Inf=1.1 --da_rot=False --da_loc_rad=2 --obs_partial=0.75
# python src/evaluation/search_param_traditional_da.py --da_method='LETKF' --da_N=20 --da_Inf=1.1 --da_rot=True --da_loc_rad=4 --obs_partial=0.75
# python src/evaluation/search_param_traditional_da.py --da_method='LETKF' --da_N=20 --da_Inf=1.1 --da_rot=False --da_loc_rad=4 --obs_partial=0.75
# python src/evaluation/search_param_traditional_da.py --da_method='LETKF' --da_N=20 --da_Inf=1.1 --da_rot=True --da_loc_rad=6 --obs_partial=0.75
# python src/evaluation/search_param_traditional_da.py --da_method='LETKF' --da_N=20 --da_Inf=1.1 --da_rot=Fasle --da_loc_rad=6 --obs_partial=0.75
# python src/evaluation/search_param_traditional_da.py --da_method='LETKF' --da_N=20 --da_Inf=1.1 --da_rot=True --da_loc_rad=8 --obs_partial=0.75
# python src/evaluation/search_param_traditional_da.py --da_method='LETKF' --da_N=20 --da_Inf=1.1 --da_rot=False --da_loc_rad=8 --obs_partial=0.75
# python src/evaluation/search_param_traditional_da.py --da_method='LETKF' --da_N=20 --da_Inf=1.1 --da_rot=True --da_loc_rad=10 --obs_partial=0.75
# python src/evaluation/search_param_traditional_da.py --da_method='LETKF' --da_N=20 --da_Inf=1.1 --da_rot=False --da_loc_rad=10 --obs_partial=0.75
## python src/evaluation/search_param_traditional_da.py --da_method='LETKF' --da_N=20 --da_Inf=1.1 --da_rot=True --da_loc_rad=6 --obs_partial=0.75
## python src/evaluation/search_param_traditional_da.py --da_method='LETKF' --da_N=20 --da_Inf=1.1 --da_rot=False --da_loc_rad=6 --obs_partial=0.75
## python src/evaluation/search_param_traditional_da.py --da_method='LETKF' --da_N=20 --da_Inf=1.1 --da_rot=True --da_loc_rad=7 --obs_partial=0.75
## python src/evaluation/search_param_traditional_da.py --da_method='LETKF' --da_N=20 --da_Inf=1.1 --da_rot=False --da_loc_rad=7 --obs_partial=0.75
#