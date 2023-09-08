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

#python src/data_factory/generate_dataset_L96.py --da_method='My4DVar' --da_N=20 --da_Bx=0.08 --da_Inf=1.0 --da_rot=False --da_loc_rad=1.0 --da_Lag=1  --obs_partial=1.0 --years=4

python src/evaluation/eval_cycles.py --da_model=seresnet_v2 --mlda_name=NN4DVar --da_method=My4DVar --da_N=20 --da_Bx=0.1 --da_Inf=1.0 --da_rot=False --da_loc_rad=1.0 --da_Lag=1  --obs_partial=1.0 --normalize=False --pred_len=4 --dim=100
python src/evaluation/eval_cycles.py --da_model=seresnet_v2 --mlda_name=NN4DVar --da_method=My4DVar --da_N=20 --da_Bx=0.1 --da_Inf=1.0 --da_rot=False --da_loc_rad=1.0 --da_Lag=1  --obs_partial=0.75 --normalize=False --pred_len=4 --dim=100

python src/evaluation/eval_cycles.py --da_model=seresnet_v2 --mlda_name=NN4DVar --da_method=My4DVar --da_N=20 --da_Bx=0.2 --da_Inf=1.0 --da_rot=False --da_loc_rad=1.0 --da_Lag=1  --obs_partial=1.0 --normalize=False --pred_len=4 --dim=200
python src/evaluation/eval_cycles.py --da_model=seresnet_v2 --mlda_name=NN4DVar --da_method=My4DVar --da_N=20 --da_Bx=0.4 --da_Inf=1.0 --da_rot=False --da_loc_rad=1.0 --da_Lag=1  --obs_partial=0.75 --normalize=False --pred_len=4 --dim=200

python src/evaluation/eval_cycles.py --da_model=seresnet_v2 --mlda_name=NN4DVar --da_method=My4DVar --da_N=20 --da_Bx=0.2 --da_Inf=1.0 --da_rot=False --da_loc_rad=1.0 --da_Lag=1  --obs_partial=1.0 --normalize=False --pred_len=4 --dim=300
python src/evaluation/eval_cycles.py --da_model=seresnet_v2 --mlda_name=NN4DVar --da_method=My4DVar --da_N=20 --da_Bx=0.6 --da_Inf=1.0 --da_rot=False --da_loc_rad=1.0 --da_Lag=1  --obs_partial=0.75 --normalize=False --pred_len=4 --dim=300

# python src/evaluation/eval_cycles.py --da_model=seresnet_v2 --mlda_name=NN4DVar --da_method=My4DVar --da_N=20 --da_Bx=0.2 --da_Inf=1.0 --da_rot=False --da_loc_rad=1.0 --da_Lag=1  --obs_partial=1.0 --normalize=False --pred_len=4 --dim=400
# python src/evaluation/eval_cycles.py --da_model=seresnet_v2 --mlda_name=NN4DVar --da_method=My4DVar --da_N=20 --da_Bx=1.0 --da_Inf=1.0 --da_rot=False --da_loc_rad=1.0 --da_Lag=1  --obs_partial=0.75 --normalize=False --pred_len=4 --dim=400

# python src/evaluation/eval_cycles.py --da_model=seresnet_v2 --mlda_name=NN4DVar --da_method=My4DVar --da_N=20 --da_Bx=0.8 --da_Inf=1.0 --da_rot=False --da_loc_rad=1.0 --da_Lag=1  --obs_partial=1.0 --normalize=False --pred_len=4 --dim=500
# python src/evaluation/eval_cycles.py --da_model=seresnet_v2 --mlda_name=NN4DVar --da_method=My4DVar --da_N=20 --da_Bx=4.0 --da_Inf=1.0 --da_rot=False --da_loc_rad=1.0 --da_Lag=1  --obs_partial=0.75 --normalize=False --pred_len=4 --dim=500