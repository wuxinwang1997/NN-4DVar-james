#!/bin/bash

python src/evaluation/search_param_traditional_da.py --system='QG' --da_method='My4DVar' --da_Lag=1 --da_Bx=0.001 --years=0.5
python src/evaluation/search_param_traditional_da.py --system='QG' --da_method='My4DVar' --da_Lag=1 --da_Bx=0.002 --years=0.5
python src/evaluation/search_param_traditional_da.py --system='QG' --da_method='My4DVar' --da_Lag=1 --da_Bx=0.004 --years=0.5
python src/evaluation/search_param_traditional_da.py --system='QG' --da_method='My4DVar' --da_Lag=1 --da_Bx=0.007 --years=0.5
python src/evaluation/search_param_traditional_da.py --system='QG' --da_method='My4DVar' --da_Lag=1 --da_Bx=0.01 --years=0.5
python src/evaluation/search_param_traditional_da.py --system='QG' --da_method='My4DVar' --da_Lag=1 --da_Bx=0.02 --years=0.5
python src/evaluation/search_param_traditional_da.py --system='QG' --da_method='My4DVar' --da_Lag=1 --da_Bx=0.04 --years=0.5
python src/evaluation/search_param_traditional_da.py --system='QG' --da_method='My4DVar' --da_Lag=1 --da_Bx=0.07 --years=0.5
python src/evaluation/search_param_traditional_da.py --system='QG' --da_method='My4DVar' --da_Lag=1 --da_Bx=0.1 --years=0.5
python src/evaluation/search_param_traditional_da.py --system='QG' --da_method='My4DVar' --da_Lag=1 --da_Bx=0.2 --years=0.5
python src/evaluation/search_param_traditional_da.py --system='QG' --da_method='My4DVar' --da_Lag=1 --da_Bx=0.4 --years=0.5
python src/evaluation/search_param_traditional_da.py --system='QG' --da_method='My4DVar' --da_Lag=1 --da_Bx=0.7 --years=0.5
python src/evaluation/search_param_traditional_da.py --system='QG' --da_method='My4DVar' --da_Lag=1 --da_Bx=1.0 --years=0.5

python src/evaluation/search_param_traditional_da.py --system='QG' --da_method='EnKF' --da_N=25 --da_Inf=1.0 --da_rot=True --years=0.5
python src/evaluation/search_param_traditional_da.py --system='QG' --da_method='EnKF' --da_N=25 --da_Inf=1.0 --da_rot=Flase --years=0.5
python src/evaluation/search_param_traditional_da.py --system='QG' --da_method='EnKF' --da_N=25 --da_Inf=1.01 --da_rot=True --years=0.5
python src/evaluation/search_param_traditional_da.py --system='QG' --da_method='EnKF' --da_N=25 --da_Inf=1.01 --da_rot=False --years=0.5
python src/evaluation/search_param_traditional_da.py --system='QG' --da_method='EnKF' --da_N=25 --da_Inf=1.02 --da_rot=True --years=0.5
python src/evaluation/search_param_traditional_da.py --system='QG' --da_method='EnKF' --da_N=25 --da_Inf=1.02 --da_rot=False --years=0.5
python src/evaluation/search_param_traditional_da.py --system='QG' --da_method='EnKF' --da_N=25 --da_Inf=1.04 --da_rot=True --years=0.5
python src/evaluation/search_param_traditional_da.py --system='QG' --da_method='EnKF' --da_N=25 --da_Inf=1.04 --da_rot=False --years=0.5
python src/evaluation/search_param_traditional_da.py --system='QG' --da_method='EnKF' --da_N=25 --da_Inf=1.07 --da_rot=True --years=0.5
python src/evaluation/search_param_traditional_da.py --system='QG' --da_method='EnKF' --da_N=25 --da_Inf=1.07 --da_rot=False --years=0.5
python src/evaluation/search_param_traditional_da.py --system='QG' --da_method='EnKF' --da_N=25 --da_Inf=1.1 --da_rot=True --years=0.5
python src/evaluation/search_param_traditional_da.py --system='QG' --da_method='EnKF' --da_N=25 --da_Inf=1.1 --da_rot=False --years=0.5

python src/evaluation/search_param_traditional_da.py --system='QG' --da_method='iEnKS' --da_N=25 --da_Inf=1.0 --da_rot=True --years=0.5
python src/evaluation/search_param_traditional_da.py --system='QG' --da_method='iEnKS' --da_N=25 --da_Inf=1.0 --da_rot=Flase --years=0.5
python src/evaluation/search_param_traditional_da.py --system='QG' --da_method='iEnKS' --da_N=25 --da_Inf=1.01 --da_rot=True --years=0.5
python src/evaluation/search_param_traditional_da.py --system='QG' --da_method='iEnKS' --da_N=25 --da_Inf=1.01 --da_rot=False --years=0.5
python src/evaluation/search_param_traditional_da.py --system='QG' --da_method='iEnKS' --da_N=25 --da_Inf=1.02 --da_rot=True --years=0.5
python src/evaluation/search_param_traditional_da.py --system='QG' --da_method='iEnKS' --da_N=25 --da_Inf=1.02 --da_rot=False --years=0.5
python src/evaluation/search_param_traditional_da.py --system='QG' --da_method='iEnKS' --da_N=25 --da_Inf=1.04 --da_rot=True --years=0.5
python src/evaluation/search_param_traditional_da.py --system='QG' --da_method='iEnKS' --da_N=25 --da_Inf=1.04 --da_rot=False --years=0.5
python src/evaluation/search_param_traditional_da.py --system='QG' --da_method='iEnKS' --da_N=25 --da_Inf=1.07 --da_rot=True --years=0.5
python src/evaluation/search_param_traditional_da.py --system='QG' --da_method='iEnKS' --da_N=25 --da_Inf=1.07 --da_rot=False --years=0.5
python src/evaluation/search_param_traditional_da.py --system='QG' --da_method='iEnKS' --da_N=25 --da_Inf=1.1 --da_rot=True --years=0.5
python src/evaluation/search_param_traditional_da.py --system='QG' --da_method='iEnKS' --da_N=25 --da_Inf=1.1 --da_rot=False --years=0.5

python src/evaluation/search_param_traditional_da.py --system='QG' --da_method='LETKF' --da_N=25 --da_Inf=1.0 --da_rot=True --da_loc_rad=1 --years=0.5
python src/evaluation/search_param_traditional_da.py --system='QG' --da_method='LETKF' --da_N=25 --da_Inf=1.0 --da_rot=False --da_loc_rad=1 --years=0.5
python src/evaluation/search_param_traditional_da.py --system='QG' --da_method='LETKF' --da_N=25 --da_Inf=1.0 --da_rot=True --da_loc_rad=2 --years=0.5
python src/evaluation/search_param_traditional_da.py --system='QG' --da_method='LETKF' --da_N=25 --da_Inf=1.0 --da_rot=False --da_loc_rad=2 --years=0.5
python src/evaluation/search_param_traditional_da.py --system='QG' --da_method='LETKF' --da_N=25 --da_Inf=1.0 --da_rot=True --da_loc_rad=3 --years=0.5
python src/evaluation/search_param_traditional_da.py --system='QG' --da_method='LETKF' --da_N=25 --da_Inf=1.0 --da_rot=False --da_loc_rad=3 --years=0.5
python src/evaluation/search_param_traditional_da.py --system='QG' --da_method='LETKF' --da_N=25 --da_Inf=1.0 --da_rot=True --da_loc_rad=4 --years=0.5
python src/evaluation/search_param_traditional_da.py --system='QG' --da_method='LETKF' --da_N=25 --da_Inf=1.0 --da_rot=False --da_loc_rad=4 --years=0.5
python src/evaluation/search_param_traditional_da.py --system='QG' --da_method='LETKF' --da_N=25 --da_Inf=1.0 --da_rot=True --da_loc_rad=5 --years=0.5
python src/evaluation/search_param_traditional_da.py --system='QG' --da_method='LETKF' --da_N=25 --da_Inf=1.0 --da_rot=False --da_loc_rad=5 --years=0.5
python src/evaluation/search_param_traditional_da.py --system='QG' --da_method='LETKF' --da_N=25 --da_Inf=1.0 --da_rot=True --da_loc_rad=6 --years=0.5
python src/evaluation/search_param_traditional_da.py --system='QG' --da_method='LETKF' --da_N=25 --da_Inf=1.0 --da_rot=False --da_loc_rad=6 --years=0.5
python src/evaluation/search_param_traditional_da.py --system='QG' --da_method='LETKF' --da_N=25 --da_Inf=1.0 --da_rot=True --da_loc_rad=7 --years=0.5
python src/evaluation/search_param_traditional_da.py --system='QG' --da_method='LETKF' --da_N=25 --da_Inf=1.0 --da_rot=False --da_loc_rad=7 --years=0.5
python src/evaluation/search_param_traditional_da.py --system='QG' --da_method='LETKF' --da_N=25 --da_Inf=1.01 --da_rot=True --da_loc_rad=1 --years=0.5
python src/evaluation/search_param_traditional_da.py --system='QG' --da_method='LETKF' --da_N=25 --da_Inf=1.01 --da_rot=False --da_loc_rad=1 --years=0.5
python src/evaluation/search_param_traditional_da.py --system='QG' --da_method='LETKF' --da_N=25 --da_Inf=1.01 --da_rot=True --da_loc_rad=2 --years=0.5
python src/evaluation/search_param_traditional_da.py --system='QG' --da_method='LETKF' --da_N=25 --da_Inf=1.01 --da_rot=False --da_loc_rad=2 --years=0.5
python src/evaluation/search_param_traditional_da.py --system='QG' --da_method='LETKF' --da_N=25 --da_Inf=1.01 --da_rot=True --da_loc_rad=3 --years=0.5
python src/evaluation/search_param_traditional_da.py --system='QG' --da_method='LETKF' --da_N=25 --da_Inf=1.01 --da_rot=False --da_loc_rad=3 --years=0.5
python src/evaluation/search_param_traditional_da.py --system='QG' --da_method='LETKF' --da_N=25 --da_Inf=1.01 --da_rot=True --da_loc_rad=4 --years=0.5
python src/evaluation/search_param_traditional_da.py --system='QG' --da_method='LETKF' --da_N=25 --da_Inf=1.01 --da_rot=False --da_loc_rad=4 --years=0.5
python src/evaluation/search_param_traditional_da.py --system='QG' --da_method='LETKF' --da_N=25 --da_Inf=1.01 --da_rot=True --da_loc_rad=5 --years=0.5
python src/evaluation/search_param_traditional_da.py --system='QG' --da_method='LETKF' --da_N=25 --da_Inf=1.01 --da_rot=False --da_loc_rad=5 --years=0.5
python src/evaluation/search_param_traditional_da.py --system='QG' --da_method='LETKF' --da_N=25 --da_Inf=1.01 --da_rot=True --da_loc_rad=6 --years=0.5
python src/evaluation/search_param_traditional_da.py --system='QG' --da_method='LETKF' --da_N=25 --da_Inf=1.01 --da_rot=False --da_loc_rad=6 --years=0.5
python src/evaluation/search_param_traditional_da.py --system='QG' --da_method='LETKF' --da_N=25 --da_Inf=1.01 --da_rot=True --da_loc_rad=7 --years=0.5
python src/evaluation/search_param_traditional_da.py --system='QG' --da_method='LETKF' --da_N=25 --da_Inf=1.01 --da_rot=False --da_loc_rad=7 --years=0.5
python src/evaluation/search_param_traditional_da.py --system='QG' --da_method='LETKF' --da_N=25 --da_Inf=1.02 --da_rot=True --da_loc_rad=1 --years=0.5
python src/evaluation/search_param_traditional_da.py --system='QG' --da_method='LETKF' --da_N=25 --da_Inf=1.02 --da_rot=False --da_loc_rad=1 --years=0.5
python src/evaluation/search_param_traditional_da.py --system='QG' --da_method='LETKF' --da_N=25 --da_Inf=1.02 --da_rot=True --da_loc_rad=2 --years=0.5
python src/evaluation/search_param_traditional_da.py --system='QG' --da_method='LETKF' --da_N=25 --da_Inf=1.02 --da_rot=False --da_loc_rad=2 --years=0.5
python src/evaluation/search_param_traditional_da.py --system='QG' --da_method='LETKF' --da_N=25 --da_Inf=1.02 --da_rot=True --da_loc_rad=3 --years=0.5
python src/evaluation/search_param_traditional_da.py --system='QG' --da_method='LETKF' --da_N=25 --da_Inf=1.02 --da_rot=False --da_loc_rad=3 --years=0.5
python src/evaluation/search_param_traditional_da.py --system='QG' --da_method='LETKF' --da_N=25 --da_Inf=1.02 --da_rot=True --da_loc_rad=4 --years=0.5
python src/evaluation/search_param_traditional_da.py --system='QG' --da_method='LETKF' --da_N=25 --da_Inf=1.02 --da_rot=False --da_loc_rad=4 --years=0.5
python src/evaluation/search_param_traditional_da.py --system='QG' --da_method='LETKF' --da_N=25 --da_Inf=1.02 --da_rot=True --da_loc_rad=5 --years=0.5
python src/evaluation/search_param_traditional_da.py --system='QG' --da_method='LETKF' --da_N=25 --da_Inf=1.02 --da_rot=False --da_loc_rad=5 --years=0.5
python src/evaluation/search_param_traditional_da.py --system='QG' --da_method='LETKF' --da_N=25 --da_Inf=1.02 --da_rot=True --da_loc_rad=6 --years=0.5
python src/evaluation/search_param_traditional_da.py --system='QG' --da_method='LETKF' --da_N=25 --da_Inf=1.02 --da_rot=False --da_loc_rad=6 --years=0.5
python src/evaluation/search_param_traditional_da.py --system='QG' --da_method='LETKF' --da_N=25 --da_Inf=1.02 --da_rot=True --da_loc_rad=7 --years=0.5
python src/evaluation/search_param_traditional_da.py --system='QG' --da_method='LETKF' --da_N=25 --da_Inf=1.02 --da_rot=False --da_loc_rad=7 --years=0.5
python src/evaluation/search_param_traditional_da.py --system='QG' --da_method='LETKF' --da_N=25 --da_Inf=1.04 --da_rot=True --da_loc_rad=1 --years=0.5
python src/evaluation/search_param_traditional_da.py --system='QG' --da_method='LETKF' --da_N=25 --da_Inf=1.04 --da_rot=False --da_loc_rad=1 --years=0.5
python src/evaluation/search_param_traditional_da.py --system='QG' --da_method='LETKF' --da_N=25 --da_Inf=1.04 --da_rot=True --da_loc_rad=2 --years=0.5
python src/evaluation/search_param_traditional_da.py --system='QG' --da_method='LETKF' --da_N=25 --da_Inf=1.04 --da_rot=False --da_loc_rad=2 --years=0.5
python src/evaluation/search_param_traditional_da.py --system='QG' --da_method='LETKF' --da_N=25 --da_Inf=1.04 --da_rot=True --da_loc_rad=3 --years=0.5
python src/evaluation/search_param_traditional_da.py --system='QG' --da_method='LETKF' --da_N=25 --da_Inf=1.04 --da_rot=Fasle --da_loc_rad=3 --years=0.5
python src/evaluation/search_param_traditional_da.py --system='QG' --da_method='LETKF' --da_N=25 --da_Inf=1.04 --da_rot=True --da_loc_rad=4 --years=0.5
python src/evaluation/search_param_traditional_da.py --system='QG' --da_method='LETKF' --da_N=25 --da_Inf=1.04 --da_rot=False --da_loc_rad=4 --years=0.5
python src/evaluation/search_param_traditional_da.py --system='QG' --da_method='LETKF' --da_N=25 --da_Inf=1.04 --da_rot=True --da_loc_rad=5 --years=0.5
python src/evaluation/search_param_traditional_da.py --system='QG' --da_method='LETKF' --da_N=25 --da_Inf=1.04 --da_rot=False --da_loc_rad=5 --years=0.5
python src/evaluation/search_param_traditional_da.py --system='QG' --da_method='LETKF' --da_N=25 --da_Inf=1.04 --da_rot=True --da_loc_rad=6 --years=0.5
python src/evaluation/search_param_traditional_da.py --system='QG' --da_method='LETKF' --da_N=25 --da_Inf=1.04 --da_rot=False --da_loc_rad=6 --years=0.5
python src/evaluation/search_param_traditional_da.py --system='QG' --da_method='LETKF' --da_N=25 --da_Inf=1.04 --da_rot=True --da_loc_rad=7 --years=0.5
python src/evaluation/search_param_traditional_da.py --system='QG' --da_method='LETKF' --da_N=25 --da_Inf=1.04 --da_rot=False --da_loc_rad=7 --years=0.5
python src/evaluation/search_param_traditional_da.py --system='QG' --da_method='LETKF' --da_N=25 --da_Inf=1.07 --da_rot=True --da_loc_rad=1 --years=0.5
python src/evaluation/search_param_traditional_da.py --system='QG' --da_method='LETKF' --da_N=25 --da_Inf=1.07 --da_rot=False --da_loc_rad=1 --years=0.5
python src/evaluation/search_param_traditional_da.py --system='QG' --da_method='LETKF' --da_N=25 --da_Inf=1.07 --da_rot=True --da_loc_rad=2 --years=0.5
python src/evaluation/search_param_traditional_da.py --system='QG' --da_method='LETKF' --da_N=25 --da_Inf=1.07 --da_rot=False --da_loc_rad=2 --years=0.5
python src/evaluation/search_param_traditional_da.py --system='QG' --da_method='LETKF' --da_N=25 --da_Inf=1.07 --da_rot=True --da_loc_rad=3 --years=0.5
python src/evaluation/search_param_traditional_da.py --system='QG' --da_method='LETKF' --da_N=25 --da_Inf=1.07 --da_rot=Fasle --da_loc_rad=3 --years=0.5
python src/evaluation/search_param_traditional_da.py --system='QG' --da_method='LETKF' --da_N=25 --da_Inf=1.07 --da_rot=True --da_loc_rad=4 --years=0.5
python src/evaluation/search_param_traditional_da.py --system='QG' --da_method='LETKF' --da_N=25 --da_Inf=1.07 --da_rot=False --da_loc_rad=4 --years=0.5
python src/evaluation/search_param_traditional_da.py --system='QG' --da_method='LETKF' --da_N=25 --da_Inf=1.07 --da_rot=True --da_loc_rad=5 --years=0.5
python src/evaluation/search_param_traditional_da.py --system='QG' --da_method='LETKF' --da_N=25 --da_Inf=1.07 --da_rot=False --da_loc_rad=5 --years=0.5
python src/evaluation/search_param_traditional_da.py --system='QG' --da_method='LETKF' --da_N=25 --da_Inf=1.07 --da_rot=True --da_loc_rad=6 --years=0.5
python src/evaluation/search_param_traditional_da.py --system='QG' --da_method='LETKF' --da_N=25 --da_Inf=1.07 --da_rot=False --da_loc_rad=6 --years=0.5
python src/evaluation/search_param_traditional_da.py --system='QG' --da_method='LETKF' --da_N=25 --da_Inf=1.07 --da_rot=True --da_loc_rad=7 --years=0.5
python src/evaluation/search_param_traditional_da.py --system='QG' --da_method='LETKF' --da_N=25 --da_Inf=1.07 --da_rot=False --da_loc_rad=7 --years=0.5
python src/evaluation/search_param_traditional_da.py --system='QG' --da_method='LETKF' --da_N=25 --da_Inf=1.1 --da_rot=True --da_loc_rad=1 --years=0.5
python src/evaluation/search_param_traditional_da.py --system='QG' --da_method='LETKF' --da_N=25 --da_Inf=1.1 --da_rot=False --da_loc_rad=1 --years=0.5
python src/evaluation/search_param_traditional_da.py --system='QG' --da_method='LETKF' --da_N=25 --da_Inf=1.1 --da_rot=True --da_loc_rad=2 --years=0.5
python src/evaluation/search_param_traditional_da.py --system='QG' --da_method='LETKF' --da_N=25 --da_Inf=1.1 --da_rot=False --da_loc_rad=2 --years=0.5
python src/evaluation/search_param_traditional_da.py --system='QG' --da_method='LETKF' --da_N=25 --da_Inf=1.1 --da_rot=True --da_loc_rad=3 --years=0.5
python src/evaluation/search_param_traditional_da.py --system='QG' --da_method='LETKF' --da_N=25 --da_Inf=1.1 --da_rot=Fasle --da_loc_rad=3 --years=0.5
python src/evaluation/search_param_traditional_da.py --system='QG' --da_method='LETKF' --da_N=25 --da_Inf=1.1 --da_rot=True --da_loc_rad=4 --years=0.5
python src/evaluation/search_param_traditional_da.py --system='QG' --da_method='LETKF' --da_N=25 --da_Inf=1.1 --da_rot=False --da_loc_rad=4 --years=0.5
python src/evaluation/search_param_traditional_da.py --system='QG' --da_method='LETKF' --da_N=25 --da_Inf=1.1 --da_rot=True --da_loc_rad=5 --years=0.5
python src/evaluation/search_param_traditional_da.py --system='QG' --da_method='LETKF' --da_N=25 --da_Inf=1.1 --da_rot=False --da_loc_rad=5 --years=0.5
python src/evaluation/search_param_traditional_da.py --system='QG' --da_method='LETKF' --da_N=25 --da_Inf=1.1 --da_rot=True --da_loc_rad=6 --years=0.5
python src/evaluation/search_param_traditional_da.py --system='QG' --da_method='LETKF' --da_N=25 --da_Inf=1.1 --da_rot=False --da_loc_rad=6 --years=0.5
python src/evaluation/search_param_traditional_da.py --system='QG' --da_method='LETKF' --da_N=25 --da_Inf=1.1 --da_rot=True --da_loc_rad=7 --years=0.5
python src/evaluation/search_param_traditional_da.py --system='QG' --da_method='LETKF' --da_N=25 --da_Inf=1.1 --da_rot=False --da_loc_rad=7 --years=0.5








