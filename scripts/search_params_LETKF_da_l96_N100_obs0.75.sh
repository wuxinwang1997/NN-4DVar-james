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
#SBATCH --output=./slurmlogs/search_params_for_L96_IEnKS_with_100observations-%j.out
#SBATCH --error=./slurmlogs/search_params_for_L96_IEnKS_with_100observations-%j.err

#python src/evaluation/search_param_traditional_da.py --da_method='Var4D' --da_Lag=1 --da_Bx=0.002 --obs_partial=0.75
#python src/evaluation/search_param_traditional_da.py --da_method='Var4D' --da_Lag=1 --da_Bx=0.004 --obs_partial=0.75
#python src/evaluation/search_param_traditional_da.py --da_method='Var4D' --da_Lag=1 --da_Bx=0.006 --obs_partial=0.75
#python src/evaluation/search_param_traditional_da.py --da_method='Var4D' --da_Lag=1 --da_Bx=0.008 --obs_partial=0.75
#python src/evaluation/search_param_traditional_da.py --da_method='Var4D' --da_Lag=1 --da_Bx=0.02 --obs_partial=0.75
#python src/evaluation/search_param_traditional_da.py --da_method='Var4D' --da_Lag=1 --da_Bx=0.04 --obs_partial=0.75
#python src/evaluation/search_param_traditional_da.py --da_method='Var4D' --da_Lag=1 --da_Bx=0.06 --obs_partial=0.75
#python src/evaluation/search_param_traditional_da.py --da_method='Var4D' --da_Lag=1 --da_Bx=0.08 --obs_partial=0.75
#python src/evaluation/search_param_traditional_da.py --da_method='Var4D' --da_Lag=1 --da_Bx=0.2 --obs_partial=0.75
#python src/evaluation/search_param_traditional_da.py --da_method='Var4D' --da_Lag=1 --da_Bx=0.4 --obs_partial=0.75
#python src/evaluation/search_param_traditional_da.py --da_method='Var4D' --da_Lag=1 --da_Bx=0.5 --obs_partial=0.75
#python src/evaluation/search_param_traditional_da.py --da_method='Var4D' --da_Lag=1 --da_Bx=0.8 --obs_partial=0.75
#python src/evaluation/search_param_traditional_da.py --da_method='Var4D' --da_Lag=1 --da_Bx=1.0 --obs_partial=0.75

# python src/evaluation/search_param_traditional_da.py --da_method='My4DVar' --da_Lag=1 --da_Bx=0.02 --obs_partial=0.75 --dim=400 --years=0.5
# python src/evaluation/search_param_traditional_da.py --da_method='My4DVar' --da_Lag=1 --da_Bx=0.04 --obs_partial=0.75 --dim=400 --years=0.5
# python src/evaluation/search_param_traditional_da.py --da_method='My4DVar' --da_Lag=1 --da_Bx=0.06 --obs_partial=0.75 --dim=400 --years=0.5
# python src/evaluation/search_param_traditional_da.py --da_method='My4DVar' --da_Lag=1 --da_Bx=0.08 --obs_partial=0.75 --dim=400 --years=0.5
# python src/evaluation/search_param_traditional_da.py --da_method='My4DVar' --da_Lag=1 --da_Bx=0.1 --obs_partial=0.75 --dim=400 --years=0.5
# python src/evaluation/search_param_traditional_da.py --da_method='My4DVar' --da_Lag=1 --da_Bx=0.2 --obs_partial=0.75 --dim=400 --years=0.5
# python src/evaluation/search_param_traditional_da.py --da_method='My4DVar' --da_Lag=1 --da_Bx=0.4 --obs_partial=0.75 --dim=400 --years=0.5
# python src/evaluation/search_param_traditional_da.py --da_method='My4DVar' --da_Lag=1 --da_Bx=0.6 --obs_partial=0.75 --dim=400 --years=0.5
# python src/evaluation/search_param_traditional_da.py --da_method='My4DVar' --da_Lag=1 --da_Bx=0.8 --obs_partial=0.75 --dim=400 --years=0.5
# python src/evaluation/search_param_traditional_da.py --da_method='My4DVar' --da_Lag=1 --da_Bx=1.0 --obs_partial=0.75 --dim=400 --years=0.5
# python src/evaluation/search_param_traditional_da.py --da_method='My4DVar' --da_Lag=1 --da_Bx=2.0 --obs_partial=0.75
# python src/evaluation/search_param_traditional_da.py --da_method='My4DVar' --da_Lag=1 --da_Bx=4.0 --obs_partial=0.75
# python src/evaluation/search_param_traditional_da.py --da_method='My4DVar' --da_Lag=1 --da_Bx=6.0 --obs_partial=0.75
# python src/evaluation/search_param_traditional_da.py --da_method='My4DVar' --da_Lag=1 --da_Bx=8.0 --obs_partial=0.75
# python src/evaluation/search_param_traditional_da.py --da_method='My4DVar' --da_Lag=1 --da_Bx=1.2 --obs_partial=0.75
# python src/evaluation/search_param_traditional_da.py --da_method='My4DVar' --da_Lag=1 --da_Bx=1.4 --obs_partial=0.75
# python src/evaluation/search_param_traditional_da.py --da_method='My4DVar' --da_Lag=1 --da_Bx=1.6 --obs_partial=0.75
# python src/evaluation/search_param_traditional_da.py --da_method='My4DVar' --da_Lag=1 --da_Bx=1.8 --obs_partial=0.75

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
# python src/evaluation/search_param_traditional_da.py --da_method='LETKF' --da_N=20 --da_Inf=1.0 --da_rot=True --da_loc_rad=2 --obs_partial=0.75 --dim=100 --years=0.5
# python src/evaluation/search_param_traditional_da.py --da_method='LETKF' --da_N=20 --da_Inf=1.0 --da_rot=False --da_loc_rad=2 --obs_partial=0.75 --dim=100 --years=0.5
# python src/evaluation/search_param_traditional_da.py --da_method='LETKF' --da_N=20 --da_Inf=1.0 --da_rot=True --da_loc_rad=4 --obs_partial=0.75 --dim=100 --years=0.5
# python src/evaluation/search_param_traditional_da.py --da_method='LETKF' --da_N=20 --da_Inf=1.0 --da_rot=False --da_loc_rad=4 --obs_partial=0.75 --dim=100 --years=0.5
# python src/evaluation/search_param_traditional_da.py --da_method='LETKF' --da_N=20 --da_Inf=1.0 --da_rot=True --da_loc_rad=6 --obs_partial=0.75 --dim=100 --years=0.5
# python src/evaluation/search_param_traditional_da.py --da_method='LETKF' --da_N=20 --da_Inf=1.0 --da_rot=False --da_loc_rad=6 --obs_partial=0.75 --dim=100 --years=0.5
# python src/evaluation/search_param_traditional_da.py --da_method='LETKF' --da_N=20 --da_Inf=1.0 --da_rot=True --da_loc_rad=8 --obs_partial=0.75 --dim=100 --years=0.5
# python src/evaluation/search_param_traditional_da.py --da_method='LETKF' --da_N=20 --da_Inf=1.0 --da_rot=False --da_loc_rad=8 --obs_partial=0.75 --dim=100 --years=0.5
# python src/evaluation/search_param_traditional_da.py --da_method='LETKF' --da_N=20 --da_Inf=1.0 --da_rot=True --da_loc_rad=10 --obs_partial=0.75 --dim=100 --years=0.5
# python src/evaluation/search_param_traditional_da.py --da_method='LETKF' --da_N=20 --da_Inf=1.0 --da_rot=False --da_loc_rad=10 --obs_partial=0.75 --dim=100 --years=0.5
# # python src/evaluation/search_param_traditional_da.py --da_method='LETKF' --da_N=20 --da_Inf=1.0 --da_rot=True --da_loc_rad=6 --obs_partial=0.75
# # python src/evaluation/search_param_traditional_da.py --da_method='LETKF' --da_N=20 --da_Inf=1.0 --da_rot=False --da_loc_rad=6 --obs_partial=0.75
# # python src/evaluation/search_param_traditional_da.py --da_method='LETKF' --da_N=20 --da_Inf=1.0 --da_rot=True --da_loc_rad=7 --obs_partial=0.75
# # python src/evaluation/search_param_traditional_da.py --da_method='LETKF' --da_N=20 --da_Inf=1.0 --da_rot=False --da_loc_rad=7 --obs_partial=0.75
# python src/evaluation/search_param_traditional_da.py --da_method='LETKF' --da_N=20 --da_Inf=1.02 --da_rot=True --da_loc_rad=2 --obs_partial=0.75 --dim=100 --years=0.5
# python src/evaluation/search_param_traditional_da.py --da_method='LETKF' --da_N=20 --da_Inf=1.02 --da_rot=False --da_loc_rad=2 --obs_partial=0.75 --dim=100 --years=0.5
# python src/evaluation/search_param_traditional_da.py --da_method='LETKF' --da_N=20 --da_Inf=1.02 --da_rot=True --da_loc_rad=4 --obs_partial=0.75 --dim=100 --years=0.5
# python src/evaluation/search_param_traditional_da.py --da_method='LETKF' --da_N=20 --da_Inf=1.02 --da_rot=False --da_loc_rad=4 --obs_partial=0.75 --dim=100 --years=0.5
# python src/evaluation/search_param_traditional_da.py --da_method='LETKF' --da_N=20 --da_Inf=1.02 --da_rot=True --da_loc_rad=6 --obs_partial=0.75 --dim=100 --years=0.5
# python src/evaluation/search_param_traditional_da.py --da_method='LETKF' --da_N=20 --da_Inf=1.02 --da_rot=False --da_loc_rad=6 --obs_partial=0.75 --dim=100 --years=0.5
# python src/evaluation/search_param_traditional_da.py --da_method='LETKF' --da_N=20 --da_Inf=1.02 --da_rot=True --da_loc_rad=8 --obs_partial=0.75 --dim=100 --years=0.5
# python src/evaluation/search_param_traditional_da.py --da_method='LETKF' --da_N=20 --da_Inf=1.02 --da_rot=False --da_loc_rad=8 --obs_partial=0.75 --dim=100 --years=0.5
# python src/evaluation/search_param_traditional_da.py --da_method='LETKF' --da_N=20 --da_Inf=1.02 --da_rot=True --da_loc_rad=10 --obs_partial=0.75 --dim=100 --years=0.5
# python src/evaluation/search_param_traditional_da.py --da_method='LETKF' --da_N=20 --da_Inf=1.02 --da_rot=False --da_loc_rad=10 --obs_partial=0.75 --dim=100 --years=0.5
# # python src/evaluation/search_param_traditional_da.py --da_method='LETKF' --da_N=20 --da_Inf=1.02 --da_rot=True --da_loc_rad=6 --obs_partial=0.75
# # python src/evaluation/search_param_traditional_da.py --da_method='LETKF' --da_N=20 --da_Inf=1.02 --da_rot=False --da_loc_rad=6 --obs_partial=0.75
# # python src/evaluation/search_param_traditional_da.py --da_method='LETKF' --da_N=20 --da_Inf=1.02 --da_rot=True --da_loc_rad=7 --obs_partial=0.75
# # python src/evaluation/search_param_traditional_da.py --da_method='LETKF' --da_N=20 --da_Inf=1.02 --da_rot=False --da_loc_rad=7 --obs_partial=0.75
# python src/evaluation/search_param_traditional_da.py --da_method='LETKF' --da_N=20 --da_Inf=1.04 --da_rot=True --da_loc_rad=2 --obs_partial=0.75 --dim=100 --years=0.5
# python src/evaluation/search_param_traditional_da.py --da_method='LETKF' --da_N=20 --da_Inf=1.04 --da_rot=False --da_loc_rad=2 --obs_partial=0.75 --dim=100 --years=0.5
# python src/evaluation/search_param_traditional_da.py --da_method='LETKF' --da_N=20 --da_Inf=1.04 --da_rot=True --da_loc_rad=4 --obs_partial=0.75 --dim=100 --years=0.5
# python src/evaluation/search_param_traditional_da.py --da_method='LETKF' --da_N=20 --da_Inf=1.04 --da_rot=False --da_loc_rad=4 --obs_partial=0.75 --dim=100 --years=0.5
# python src/evaluation/search_param_traditional_da.py --da_method='LETKF' --da_N=20 --da_Inf=1.04 --da_rot=True --da_loc_rad=6 --obs_partial=0.75 --dim=100 --years=0.5
# python src/evaluation/search_param_traditional_da.py --da_method='LETKF' --da_N=20 --da_Inf=1.04 --da_rot=False --da_loc_rad=6 --obs_partial=0.75 --dim=100 --years=0.5
# python src/evaluation/search_param_traditional_da.py --da_method='LETKF' --da_N=20 --da_Inf=1.04 --da_rot=True --da_loc_rad=8 --obs_partial=0.75 --dim=100 --years=0.5
# python src/evaluation/search_param_traditional_da.py --da_method='LETKF' --da_N=20 --da_Inf=1.04 --da_rot=False --da_loc_rad=8 --obs_partial=0.75 --dim=100 --years=0.5
# python src/evaluation/search_param_traditional_da.py --da_method='LETKF' --da_N=20 --da_Inf=1.04 --da_rot=True --da_loc_rad=10 --obs_partial=0.75 --dim=100 --years=0.5
# python src/evaluation/search_param_traditional_da.py --da_method='LETKF' --da_N=20 --da_Inf=1.04 --da_rot=False --da_loc_rad=10 --obs_partial=0.75 --dim=100 --years=0.5
# # python src/evaluation/search_param_traditional_da.py --da_method='LETKF' --da_N=20 --da_Inf=1.04 --da_rot=True --da_loc_rad=6 --obs_partial=0.75
# # python src/evaluation/search_param_traditional_da.py --da_method='LETKF' --da_N=20 --da_Inf=1.04 --da_rot=False --da_loc_rad=6 --obs_partial=0.75
# # python src/evaluation/search_param_traditional_da.py --da_method='LETKF' --da_N=20 --da_Inf=1.04 --da_rot=True --da_loc_rad=7 --obs_partial=0.75
# # python src/evaluation/search_param_traditional_da.py --da_method='LETKF' --da_N=20 --da_Inf=1.04 --da_rot=False --da_loc_rad=7 --obs_partial=0.75
# python src/evaluation/search_param_traditional_da.py --da_method='LETKF' --da_N=20 --da_Inf=1.06 --da_rot=True --da_loc_rad=2 --obs_partial=0.75 --dim=100 --years=0.5
# python src/evaluation/search_param_traditional_da.py --da_method='LETKF' --da_N=20 --da_Inf=1.06 --da_rot=False --da_loc_rad=2 --obs_partial=0.75 --dim=100 --years=0.5
# python src/evaluation/search_param_traditional_da.py --da_method='LETKF' --da_N=20 --da_Inf=1.06 --da_rot=True --da_loc_rad=4 --obs_partial=0.75 --dim=100 --years=0.5
# python src/evaluation/search_param_traditional_da.py --da_method='LETKF' --da_N=20 --da_Inf=1.06 --da_rot=False --da_loc_rad=4 --obs_partial=0.75 --dim=100 --years=0.5
# python src/evaluation/search_param_traditional_da.py --da_method='LETKF' --da_N=20 --da_Inf=1.06 --da_rot=True --da_loc_rad=6 --obs_partial=0.75 --dim=100 --years=0.5
python src/evaluation/search_param_traditional_da.py --da_method='LETKF' --da_N=20 --da_Inf=1.06 --da_rot=False --da_loc_rad=6 --obs_partial=0.75 --dim=100 --years=0.5
# python src/evaluation/search_param_traditional_da.py --da_method='LETKF' --da_N=20 --da_Inf=1.06 --da_rot=True --da_loc_rad=8 --obs_partial=0.75 --dim=100 --years=0.5
# python src/evaluation/search_param_traditional_da.py --da_method='LETKF' --da_N=20 --da_Inf=1.06 --da_rot=False --da_loc_rad=8 --obs_partial=0.75 --dim=100 --years=0.5
# python src/evaluation/search_param_traditional_da.py --da_method='LETKF' --da_N=20 --da_Inf=1.06 --da_rot=True --da_loc_rad=10 --obs_partial=0.75 --dim=100 --years=0.5
# python src/evaluation/search_param_traditional_da.py --da_method='LETKF' --da_N=20 --da_Inf=1.06 --da_rot=False --da_loc_rad=10 --obs_partial=0.75 --dim=100 --years=0.5
# # python src/evaluation/search_param_traditional_da.py --da_method='LETKF' --da_N=20 --da_Inf=1.06 --da_rot=True --da_loc_rad=6 --obs_partial=0.75
# # python src/evaluation/search_param_traditional_da.py --da_method='LETKF' --da_N=20 --da_Inf=1.06 --da_rot=False --da_loc_rad=6 --obs_partial=0.75
# # python src/evaluation/search_param_traditional_da.py --da_method='LETKF' --da_N=20 --da_Inf=1.06 --da_rot=True --da_loc_rad=7 --obs_partial=0.75
# # python src/evaluation/search_param_traditional_da.py --da_method='LETKF' --da_N=20 --da_Inf=1.06 --da_rot=False --da_loc_rad=7 --obs_partial=0.75
# python src/evaluation/search_param_traditional_da.py --da_method='LETKF' --da_N=20 --da_Inf=1.08 --da_rot=True --da_loc_rad=2 --obs_partial=0.75 --dim=100 --years=0.5
# python src/evaluation/search_param_traditional_da.py --da_method='LETKF' --da_N=20 --da_Inf=1.08 --da_rot=False --da_loc_rad=2 --obs_partial=0.75 --dim=100 --years=0.5
# python src/evaluation/search_param_traditional_da.py --da_method='LETKF' --da_N=20 --da_Inf=1.08 --da_rot=True --da_loc_rad=4 --obs_partial=0.75 --dim=100 --years=0.5
# python src/evaluation/search_param_traditional_da.py --da_method='LETKF' --da_N=20 --da_Inf=1.08 --da_rot=False --da_loc_rad=4 --obs_partial=0.75 --dim=100 --years=0.5
# python src/evaluation/search_param_traditional_da.py --da_method='LETKF' --da_N=20 --da_Inf=1.08 --da_rot=True --da_loc_rad=6 --obs_partial=0.75 --dim=100 --years=0.5
python src/evaluation/search_param_traditional_da.py --da_method='LETKF' --da_N=20 --da_Inf=1.08 --da_rot=False --da_loc_rad=6 --obs_partial=0.75 --dim=100 --years=0.5
# python src/evaluation/search_param_traditional_da.py --da_method='LETKF' --da_N=20 --da_Inf=1.08 --da_rot=True --da_loc_rad=8 --obs_partial=0.75 --dim=100 --years=0.5
# python src/evaluation/search_param_traditional_da.py --da_method='LETKF' --da_N=20 --da_Inf=1.08 --da_rot=False --da_loc_rad=8 --obs_partial=0.75 --dim=100 --years=0.5
# python src/evaluation/search_param_traditional_da.py --da_method='LETKF' --da_N=20 --da_Inf=1.08 --da_rot=True --da_loc_rad=10 --obs_partial=0.75 --dim=100 --years=0.5
# python src/evaluation/search_param_traditional_da.py --da_method='LETKF' --da_N=20 --da_Inf=1.08 --da_rot=False --da_loc_rad=10 --obs_partial=0.75 --dim=100 --years=0.5
# # python src/evaluation/search_param_traditional_da.py --da_method='LETKF' --da_N=20 --da_Inf=1.08 --da_rot=True --da_loc_rad=6 --obs_partial=0.75
# # python src/evaluation/search_param_traditional_da.py --da_method='LETKF' --da_N=20 --da_Inf=1.08 --da_rot=False --da_loc_rad=6 --obs_partial=0.75
# # python src/evaluation/search_param_traditional_da.py --da_method='LETKF' --da_N=20 --da_Inf=1.08 --da_rot=True --da_loc_rad=7 --obs_partial=0.75
# # python src/evaluation/search_param_traditional_da.py --da_method='LETKF' --da_N=20 --da_Inf=1.08 --da_rot=False --da_loc_rad=7 --obs_partial=0.75
# python src/evaluation/search_param_traditional_da.py --da_method='LETKF' --da_N=20 --da_Inf=1.1 --da_rot=True --da_loc_rad=2 --obs_partial=0.75 --dim=100 --years=0.5
# python src/evaluation/search_param_traditional_da.py --da_method='LETKF' --da_N=20 --da_Inf=1.1 --da_rot=False --da_loc_rad=2 --obs_partial=0.75 --dim=100 --years=0.5
# python src/evaluation/search_param_traditional_da.py --da_method='LETKF' --da_N=20 --da_Inf=1.1 --da_rot=True --da_loc_rad=4 --obs_partial=0.75 --dim=100 --years=0.5
# python src/evaluation/search_param_traditional_da.py --da_method='LETKF' --da_N=20 --da_Inf=1.1 --da_rot=False --da_loc_rad=4 --obs_partial=0.75 --dim=100 --years=0.5
# python src/evaluation/search_param_traditional_da.py --da_method='LETKF' --da_N=20 --da_Inf=1.1 --da_rot=True --da_loc_rad=6 --obs_partial=0.75 --dim=100 --years=0.5
python src/evaluation/search_param_traditional_da.py --da_method='LETKF' --da_N=20 --da_Inf=1.1 --da_rot=False --da_loc_rad=6 --obs_partial=0.75 --dim=100 --years=0.5
# python src/evaluation/search_param_traditional_da.py --da_method='LETKF' --da_N=20 --da_Inf=1.1 --da_rot=True --da_loc_rad=8 --obs_partial=0.75 --dim=100 --years=0.5
# python src/evaluation/search_param_traditional_da.py --da_method='LETKF' --da_N=20 --da_Inf=1.1 --da_rot=False --da_loc_rad=8 --obs_partial=0.75 --dim=100 --years=0.5
# python src/evaluation/search_param_traditional_da.py --da_method='LETKF' --da_N=20 --da_Inf=1.1 --da_rot=True --da_loc_rad=10 --obs_partial=0.75 --dim=100 --years=0.5
# python src/evaluation/search_param_traditional_da.py --da_method='LETKF' --da_N=20 --da_Inf=1.1 --da_rot=False --da_loc_rad=10 --obs_partial=0.75 --dim=100 --years=0.5
# # python src/evaluation/search_param_traditional_da.py --da_method='LETKF' --da_N=20 --da_Inf=1.1 --da_rot=True --da_loc_rad=6 --obs_partial=0.75
# # python src/evaluation/search_param_traditional_da.py --da_method='LETKF' --da_N=20 --da_Inf=1.1 --da_rot=False --da_loc_rad=6 --obs_partial=0.75
# # python src/evaluation/search_param_traditional_da.py --da_method='LETKF' --da_N=20 --da_Inf=1.1 --da_rot=True --da_loc_rad=7 --obs_partial=0.75
# # python src/evaluation/search_param_traditional_da.py --da_method='LETKF' --da_N=20 --da_Inf=1.1 --da_rot=False --da_loc_rad=7 --obs_partial=0.75
##
