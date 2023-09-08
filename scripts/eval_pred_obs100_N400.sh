#!/bin/bash

# SLURM SUBMIT SCRIPT
#SBATCH --nodes=1
#SBATCH -p normal
#SBATCH --time=30:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=10G
#SBATCH --output=./slurmlogs/log-evalcycle-enkf-N40-partial1.0-wonorm-predlen0-%j.out
#SBATCH --error=./slurmlogs/log-evalcycle-enkf-N40-partial1.0-wonorm-predlen0-%j.err

# Schedule execution of many runs
# Run from root folder with: bash scripts/schedule.sh

#python src/data_factory/generate_dataset_L96.py --da_method=EnKF --da_N=20 --da_Inf=1.02 --da_rot=True  --obs_partial=1.0 --normalize=True --years=4

python src/evaluation/eval_pred.py --da_model=4dvarnet_ae --mlda_name=4DVarNet --da_method=My4DVar --da_N=20 --da_Bx=0.4 --da_Inf=1.0 --da_rot=False  --obs_partial=1.0 --normalize=False --da_Lag=1 --pred_len=4 --years=1 --dim=400
# python src/evaluation/eval_pred.py --da_model=seresnet_v2 --mlda_name=NN4DVar --da_method=My4DVar --da_N=20 --da_Bx=0.2 --da_Inf=1.0 --da_rot=False  --obs_partial=1.0 --normalize=True --da_Lag=1 --pred_len=4 --years=1 --dim=400

# python src/evaluation/eval_pred.py --da_model=4dvarnet_ae --mlda_name=4DVarNet --da_method=My4DVar --da_N=20 --da_Bx=0.8 --da_Inf=1.0 --da_rot=False  --obs_partial=0.75 --normalize=False --da_Lag=1 --pred_len=4 --years=1 --dim=400
# python src/evaluation/eval_pred.py --da_model=seresnet_v2 --mlda_name=NN4DVar --da_method=My4DVar --da_N=20 --da_Bx=1.0 --da_Inf=1.0 --da_rot=False --obs_partial=0.75 --normalize=True --da_Lag=1 --pred_len=4 --years=1 --dim=400

# python src/evaluation/eval_pred.py --da_model=4dvarnet_ae --mlda_name=4DVarNet --da_method=iEnKS --da_N=20 --da_Inf=1.02 --da_rot=True  --da_Lag=1 --obs_partial=1.0 --normalize=False --pred_len=0 --years=1
# python src/evaluation/eval_pred.py --da_model=seresnet_v2 --mlda_name=NN4DVar --da_method=iEnKS --da_N=20 --da_Inf=1.02 --da_rot=True  --da_Lag=1 --obs_partial=1.0 --normalize=True --pred_len=4 --years=1

# python src/evaluation/eval_pred.py --da_model=4dvarnet_ae --mlda_name=4DVarNet --da_method=LETKF --da_N=20 --da_Inf=1.02 --da_rot=True --da_loc_rad=10  --obs_partial=1.0 --normalize=False --pred_len=0 --years=1
# python src/evaluation/eval_pred.py --da_model=seresnet_v2 --mlda_name=NN4DVar --da_method=LETKF --da_N=20 --da_Inf=1.02 --da_rot=True --da_loc_rad=10  --obs_partial=1.0 --normalize=True --pred_len=4 --years=1