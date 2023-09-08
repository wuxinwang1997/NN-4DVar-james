#!/bin/bash

# SLURM SUBMIT SCRIPT
#SBATCH --nodes=1
#SBATCH -p normal
#SBATCH --time=30:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=1.0G
#SBATCH --output=./slurmlogs/log-evalcycle-enkf-N40-partial1.0-wonorm-predlen0-%j.out
#SBATCH --error=./slurmlogs/log-evalcycle-enkf-N40-partial1.0-wonorm-predlen0-%j.err

# Schedule execution of many runs
# Run from root folder with: bash scripts/schedule.sh

#python src/data_factory/generate_dataset_L96.py --da_method='My4DVar' --da_N=20 --da_Bx=0.04 --da_Inf=1.0 --da_rot=1.0 --da_loc_rad=1.0 --da_Lag=1  --obs_partial=0.75 --years=4

python src/evaluation/eval_obserr.py --da_model=seresnet_v2 --mlda_name=NN4DVar --da_method=EnKF --da_N=20 --da_Inf=1.02 --da_rot=False  --obs_partial=0.75 --normalize=True --pred_len=3 --obserr=1.5 
python src/evaluation/eval_obserr.py --da_model=seresnet_v2 --mlda_name=NN4DVar --da_method=EnKF --da_N=20 --da_Inf=1.02 --da_rot=False  --obs_partial=0.75 --normalize=True --pred_len=3 --obserr=2.0
python src/evaluation/eval_obserr.py --da_model=seresnet_v2 --mlda_name=NN4DVar --da_method=EnKF --da_N=20 --da_Inf=1.02 --da_rot=False  --obs_partial=0.75 --normalize=True --pred_len=3 --obserr=2.5
python src/evaluation/eval_obserr.py --da_model=seresnet_v2 --mlda_name=NN4DVar --da_method=EnKF --da_N=20 --da_Inf=1.02 --da_rot=False  --obs_partial=0.75 --normalize=True --pred_len=3 --obserr=3.0
# python src/evaluation/eval_obserr.py --da_model=4dvarnet_ae --mlda_name=4DVarNet --da_method=My4DVar --da_N=20 --da_Bx=0.04 --da_Inf=1.0 --da_rot=True --da_loc_rad=1.0 --da_Lag=1  --obs_partial=0.75 --normalize=False --obserr=1.5 --pred_len=0
# python src/evaluation/eval_obserr.py --da_model=4dvarnet_ae --mlda_name=4DVarNet --da_method=My4DVar --da_N=20 --da_Bx=0.04 --da_Inf=1.0 --da_rot=True --da_loc_rad=1.0 --da_Lag=1  --obs_partial=0.75 --normalize=False --obserr=2.0 --pred_len=0
# python src/evaluation/eval_obserr.py --da_model=4dvarnet_ae --mlda_name=4DVarNet --da_method=My4DVar --da_N=20 --da_Bx=0.04 --da_Inf=1.0 --da_rot=True --da_loc_rad=1.0 --da_Lag=1  --obs_partial=0.75 --normalize=False --obserr=2.5 --pred_len=0
# python src/evaluation/eval_obserr.py --da_model=4dvarnet_ae --mlda_name=4DVarNet --da_method=My4DVar --da_N=20 --da_Bx=0.04 --da_Inf=1.0 --da_rot=True --da_loc_rad=1.0 --da_Lag=1  --obs_partial=0.75 --normalize=False --obserr=3.0 --pred_len=0


# python src/evaluation/eval_obserr.py --da_model=seresnet_v2 --mlda_name=NN4DVar --da_method=My4DVar --da_N=20 --da_Bx=0.04 --da_Inf=1.0 --da_rot=True --da_loc_rad=1.0 --da_Lag=1  --obs_partial=0.75 --normalize=True --obserr=1.5 --pred_len=4
# python src/evaluation/eval_obserr.py --da_model=seresnet_v2 --mlda_name=NN4DVar --da_method=My4DVar --da_N=20 --da_Bx=0.04 --da_Inf=1.0 --da_rot=True --da_loc_rad=1.0 --da_Lag=1  --obs_partial=0.75 --normalize=True --obserr=2.0 --pred_len=4
# python src/evaluation/eval_obserr.py --da_model=seresnet_v2 --mlda_name=NN4DVar --da_method=My4DVar --da_N=20 --da_Bx=0.04 --da_Inf=1.0 --da_rot=True --da_loc_rad=1.0 --da_Lag=1  --obs_partial=0.75 --normalize=True --obserr=2.5 --pred_len=4
# python src/evaluation/eval_obserr.py --da_model=seresnet_v2 --mlda_name=NN4DVar --da_method=My4DVar --da_N=20 --da_Bx=0.04 --da_Inf=1.0 --da_rot=True --da_loc_rad=1.0 --da_Lag=1  --obs_partial=0.75 --normalize=True --obserr=3.0 --pred_len=4
