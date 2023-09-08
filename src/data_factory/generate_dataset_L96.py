import numpy as np
import os
import sys
sys.path.append('.')
from mpl_tools import is_notebook_or_qt as nb
import dapper as dpr
import dapper.da_methods as da
import dapper.tools.progressbar as pb
import dapper.mods as modelling
import dapper.mods.Lorenz96 as Lorenz96
from dapper.tools.localization import nd_Id_localization
import argparse
import torch
from src.da_methods.fdvar_l96 import My4DVar
from src.models.components.lorenz96 import Lorenz96_torch as L96torch
from src.models.components.lorenz96 import Lorenz96_partial_Id_Obs
from src.models.components.qg import torch_qg, LP_setup
from src.utils.utils import ens_compatible
import xarray as xr
from src.utils.utils import str2bool
np.random.seed(2023)

def simulate(self, desc='Truth & Obs'):
        """Generate synthetic truth and observations."""
        Dyn, Obs, tseq, X0 = self.Dyn, self.Obs, self.tseq, self.X0

        # Init
        xx    = np.zeros((tseq.K+1, Dyn.M))
        yy    = np.zeros((tseq.Ko+1, Obs.M))
        yy_    = np.zeros((tseq.K+1, Obs.M))

        x = X0.sample(1)
        if len(x.shape) == 2:
            x = np.squeeze(x, axis=0)
        xx[0] = x

        # Loop
        for k, ko, t, dt in pb.progbar(tseq.ticker, desc):
            x = Dyn(x, t-dt, dt)
            if isinstance(x, torch.Tensor):
                x = x.detach().numpy()
            yy_[k] = Obs(x, t) + Obs.noise.sample(1)
            if ko is not None:
                yy[ko] = yy_[k]
            xx[k] = x

        return xx, yy, yy_

def generate_da_data(data_dir, years, N_trials, dim, da_method, da_N, da_Inf, da_Bx, da_rot, da_loc_rad, da_xN, obs_partial, da_Lag=0):
    spin_up = int(2000 * 0.05)
    OneYear = 0.05 * (24/6) * 365
    # Sakov uses K=300000, BurnIn=1000*0.05
    tseq = modelling.Chronology(0.01, dto=0.05, T = spin_up + years * OneYear, Tplot=Lorenz96.Tplot, BurnIn=2*Lorenz96.Tplot)
    
    Nx = dim
    x0 = Lorenz96.x0(Nx)

    # select random observation grids
    # jj = np.zeros(shape=(Nx))
    # jj[:int(obs_partial*Nx)] = int(1)
    # np.random.shuffle(jj)
    # jj = np.asarray(np.where(jj==1)[0])
    if obs_partial == 0.75:
        jj = np.ones(shape=(Nx))
        jj[np.asarray(np.arange(1, Nx, 4))] = 0
        jj = np.asarray(np.where(jj==1)[0])
    else:
        jj = np.asarray(np.arange(Nx))
    # print(jj)

    def obs_inds(t):
        return jj

    if da_method == 'My4DVar':
        l96torch = L96torch(8)
        Dyn = {
            'M': Nx,
            'model': l96torch.forward,
            'noise': 0,
            'name': 'L96'
        }
        Obs = Lorenz96_partial_Id_Obs(Nx, obs_inds, jj)
        Obs['noise'] = 1
        Obs['localizer'] = nd_Id_localization((Nx,), (2,), jj)
    else:
        Dyn = {
            'M': Nx,
            'model': Lorenz96.step,
            'linear': Lorenz96.dstep_dx,
            'noise': 0,
        }

        # jj = np.arange(Nx) # 1 + np.arange(0, Nx, 2)
        Obs = modelling.partial_Id_Obs(Nx, jj)
        Obs['noise'] = 1
        Obs['localizer'] = nd_Id_localization((Nx,), (2,), jj)

    X0 = modelling.GaussRV(mu=x0, C=0.001)
    HMM = modelling.HiddenMarkovModel(Dyn, Obs, tseq, X0)
    HMM.liveplotters = Lorenz96.LPs(jj)

    # data assimilation method to be used to generate the dataset
    if da_method == 'OI':
        xp = da.OptInterp()
    if da_method == 'Var3D':
        xp = da.Var3D(xB=da_Bx)
    if da_method == 'Var4D':
        xp = da.Var4D(B='clim', Lag=1, nIter=12, xB=da_Bx, wtol=1e-5)
    if da_method == 'My4DVar':
        xp = My4DVar(B='clim', Lag=1, nIter=2, xB=da_Bx, lr=5e-1, max_iter=5, tolerance_grad=1e-9, tolerance_change=1e-12, history_size=50)
    if da_method == 'DEnKF':
        xp = da.EnKF('DEnKF',N=da_N, infl=da_Inf, rot=da_rot)
    if da_method == 'EnKF':
        xp = da.EnKF('Sqrt',N=da_N, infl=da_Inf, rot=da_rot)
    if da_method == 'LETKF':
        xp = da.LETKF(N=da_N , infl=da_Inf, rot=da_rot, loc_rad=da_loc_rad)
    if da_method == 'iEnKS':
        xp = da.iEnKS('Sqrt', N=da_N, Lag=da_Lag, infl=da_Inf, rot=da_rot, bundle=True) #xN=da_xN)
    
    xp.seed = 2023
    xxs, xbs, rmse_xbs, xas, rmse_xas, yys, yy_s = [], [], [], [], [], [], []
    for i in range(N_trials):
        HMM.X0 = modelling.GaussRV(mu=x0, C=0.001)
        my_HMM = HMM.copy()
        xx, yy, yy_ = simulate(my_HMM)

        # 需要实验往前一个同化窗口的观测的获取，观测单独保存数据
        if da_method == 'My4DVar':
            HMM.tseq.Ko -= 1
            xp.assimilate(HMM, xx[:-HMM.tseq.dko], yy[:-1], yy_[HMM.tseq.dko:])
        else:
            HMM.tseq.Ko -= 1
            xp.assimilate(HMM, xx[:-HMM.tseq.dko], yy[:-1], yy_[HMM.tseq.dko:])

        xa = xp.stats.mu.a
        xb = xp.stats.mu.f

        xb = xb[~np.isnan(xb).any(axis=1), :][int(spin_up / HMM.tseq.dto):]
        xa = xa[~np.isnan(xa).any(axis=1), :][int(spin_up / HMM.tseq.dto):]
        xx = xx[HMM.tseq.dko:-HMM.tseq.dko:HMM.tseq.dko][int(spin_up / HMM.tseq.dto):]
        yy = yy[int(spin_up / HMM.tseq.dto):-1]
        yy_ = yy_[HMM.tseq.dko:-1][int(spin_up/HMM.tseq.dt):]

        if i == 0:
            mean_xb, std_xb = np.mean(xb), np.std(xb)
            mean_xa, std_xa = np.mean(xa), np.std(xa)
            mean_obs, std_obs = np.mean(yy), np.std(yy)
            stats = dict()
            stats['mean_xb'] = mean_xb
            stats['std_xb'] = std_xb
            stats['mean_xa'] = mean_xa
            stats['std_xa'] = std_xa
            stats['mean_obs'] = mean_obs
            stats['std_obs'] = std_obs
            np.savez(f'{data_dir}/L96_stats_{da_method}_T{years}_N_trials{N_trials}_N{dim}_daN{da_N}_daInf{da_Inf}_daBx{da_Bx}_darot{da_rot}_dalocrad{da_loc_rad}_obspartial{obs_partial}_daLag{da_Lag}.npz', **stats)

        # rmse_xb = np.sqrt(np.mean((xb-xx[(np.arange(0, years*OneYear, HMM.tseq.dto)/HMM.tseq.dt).astype(np.int32)])**2, axis=-1))
        # rmse_xa = np.sqrt(np.mean((xa-xx[(np.arange(0, years*OneYear, HMM.tseq.dto)/HMM.tseq.dt).astype(np.int32)])**2, axis=-1))
        rmse_xb = np.sqrt(np.mean((xb - xx) ** 2,axis=-1))
        rmse_xa = np.sqrt(np.mean((xa - xx) ** 2, axis=-1))
        print('Xa RMSE: ', np.mean(rmse_xa))
        print('Xb RMSE: ', np.mean(rmse_xb))
        # print('Obs RMSE: ', np.mean((yy - xx) ** 2))

        xxs.append(xx)
        xbs.append(xb)
        rmse_xbs.append(rmse_xb)
        xas.append(xa)
        rmse_xas.append(rmse_xa)
        yys.append(yy)
        yy_s.append(yy_)

        HMM.tseq.Ko += 1

    xx = xr.DataArray(
        np.asarray(xxs),
        dims=['trial', 'lead_time', 'grid'],
        coords={
            'trial': np.arange(0, N_trials, 1).astype(np.int32),
            # 'lead_time': np.arange(0, xx.shape[0]*HMM.tseq.dt, HMM.tseq.dt).astype(np.float32),
            'lead_time': np.arange(0, years * OneYear-HMM.tseq.dto, HMM.tseq.dto).astype(np.float32),
            'grid': np.arange(Nx), 
        },
        name='x'
    )

    xb = xr.DataArray(
        np.asarray(xbs),
        dims=['trial', 'lead_time', 'grid'],
        coords={
            'trial': np.arange(0, N_trials, 1),
            'lead_time': np.arange(0, years*OneYear-HMM.tseq.dto, HMM.tseq.dto).astype(np.float32),
            'grid': np.arange(Nx), 
        },
        name='x'
    )

    rmse_xb = xr.DataArray(
        rmse_xbs,
        dims=['trial', 'lead_time'],
        coords={
            'trial': np.arange(0, N_trials, 1),
            'lead_time': np.arange(0, years*OneYear-HMM.tseq.dto, HMM.tseq.dto).astype(np.float32),
        },
        name='x'
    )

    xa = xr.DataArray(
        np.asarray(xas),
        dims=['trial', 'lead_time', 'grid'],
        coords={
            'trial': np.arange(0, N_trials, 1),
            'lead_time': np.arange(0, years*OneYear-HMM.tseq.dto, HMM.tseq.dto).astype(np.float32),
            'grid': np.arange(Nx), 
        },
        name='x'
    )

    rmse_xa = xr.DataArray(
        rmse_xas,
        dims=['trial', 'lead_time'],
        coords={
            'trial': np.arange(0, N_trials, 1),
            'lead_time': np.arange(0, years*OneYear-HMM.tseq.dto, HMM.tseq.dto).astype(np.float32),
        },
        name='x'
    )

    obs = xr.DataArray(
        np.asarray(yys),
        dims=['trial', 'lead_time', 'grid'],
        coords={
            'trial': np.arange(0, N_trials, 1),
            'lead_time': np.arange(0, years*OneYear-HMM.tseq.dto, HMM.tseq.dto).astype(np.float32),
            'grid': jj,
        },
        name='x'
    )

    obs_all = xr.DataArray(
        np.asarray(yy_s),
        dims=['trial', 'lead_time', 'grid'],
        coords={
            'trial': np.arange(0, N_trials, 1),
            'lead_time': np.arange(0, yy_s[0].shape[0]*HMM.tseq.dt, HMM.tseq.dt).astype(np.float32),
            'grid': jj, 
        },
        name='x'
    )

    if not os.path.exists(data_dir):
        os.mkdir(data_dir)

    xx.to_netcdf(f'{data_dir}/L96_gt_{da_method}_T{years}_N_trials{N_trials}_N{dim}_daN{da_N}_daInf{da_Inf}_daBx{da_Bx}_darot{da_rot}_dalocrad{da_loc_rad}_obspartial{obs_partial}_daLag{da_Lag}.nc')
    xb.to_netcdf(f'{data_dir}/L96_xb_{da_method}_T{years}_N_trials{N_trials}_N{dim}_daN{da_N}_daInf{da_Inf}_daBx{da_Bx}_darot{da_rot}_dalocrad{da_loc_rad}_obspartial{obs_partial}_daLag{da_Lag}.nc')
    rmse_xb.to_netcdf(f'{data_dir}/L96_rmse_xb_{da_method}_T{years}_N_trials{N_trials}_N{dim}_daN{da_N}_daInf{da_Inf}_daBx{da_Bx}_darot{da_rot}_dalocrad{da_loc_rad}_obspartial{obs_partial}_daLag{da_Lag}.nc')
    xa.to_netcdf(f'{data_dir}/L96_xa_{da_method}_T{years}_N_trials{N_trials}_N{dim}_daN{da_N}_daInf{da_Inf}_daBx{da_Bx}_darot{da_rot}_dalocrad{da_loc_rad}_obspartial{obs_partial}_daLag{da_Lag}.nc')
    rmse_xa.to_netcdf(f'{data_dir}/L96_rmse_xa_{da_method}_T{years}_N_trials{N_trials}_N{dim}_daN{da_N}_daInf{da_Inf}_daBx{da_Bx}_darot{da_rot}_dalocrad{da_loc_rad}_obspartial{obs_partial}_daLag{da_Lag}.nc')
    obs.to_netcdf(f'{data_dir}/L96_obs_{da_method}_T{years}_N_trials{N_trials}_N{dim}_daN{da_N}_daInf{da_Inf}_daBx{da_Bx}_darot{da_rot}_dalocrad{da_loc_rad}_obspartial{obs_partial}_daLag{da_Lag}.nc')
    obs_all.to_netcdf(f'{data_dir}/L96_obs_all_{da_method}_T{years}_N_trials{N_trials}_N{dim}_daN{da_N}_daInf{da_Inf}_daBx{da_Bx}_darot{da_rot}_dalocrad{da_loc_rad}_obspartial{obs_partial}_daLag{da_Lag}.nc')

def prepare_parser():
    parser = argparse.ArgumentParser(description='Generate Background')
    parser.add_argument(
        '--data_dir',
        type=str,
        help='path to save the dataset',
        default='./data/train/'
    )

    parser.add_argument(
        '--years',
        type=int,
        help = 'years for experiments',
        default=4
    )

    parser.add_argument(
        '--N_trials',
        type = int,
        help = 'trials for data generation',
        default=2
    )

    parser.add_argument(
        '--dim',
        type = int,
        help = 'dimension for Lorenz96 system',
        default=40
    )

    parser.add_argument(
        '--da_method',
        type=str,
        help = 'data assimilation method utilized to generate dataset',
        default='Var4D'
    )

    parser.add_argument(
        '--da_N',
        type=int,
        help = 'ensemble size for EnDA',
        default=20
    )

    parser.add_argument(
        '--da_Inf',
        type=float,
        help = 'Inflation hyperparameter for EnDA',
        default=1.0
    )

    parser.add_argument(
        '--da_Bx',
        type=float,
        help = 'Covariance multiple ratio',
        default=1.0
    )    

    parser.add_argument(
        '--da_rot',
        type=str2bool,
        help = 'Rotation hyperparameter for EnDA',
        default=False
    )        
    
    parser.add_argument(
        '--da_loc_rad',
        type=float,
        help = 'Localization ratio hyperparameter for EnDA',
        default=1.0
    )        

    parser.add_argument(
        '--da_xN',
        type=float,
        help = 'xN hyperparameter for EnDA',
        default=1.0
    ) 
    
    parser.add_argument(
        '--obs_partial',
        type=float,
        help = 'Observational grid partial hyperparameter',
        default = 1.0
    ) 

    parser.add_argument(
        '--da_Lag',
        type=int,
        help = 'Data assimilation window length',
        default=0
    )     
    
    return parser


if __name__ == '__main__':
    parser = prepare_parser()
    args = parser.parse_args()
    data_dir = args.data_dir
    years = args.years
    N_trials = args.N_trials
    dim = args.dim
    da_method = args.da_method
    da_N = args.da_N
    da_Inf = args.da_Inf
    da_Bx = args.da_Bx
    da_rot = args.da_rot
    da_loc_rad = args.da_loc_rad
    da_xN = args.da_xN
    obs_partial = args.obs_partial
    da_Lag = args.da_Lag

    generate_da_data(data_dir, years, N_trials, dim, da_method, da_N, da_Inf, da_Bx, da_rot, da_loc_rad, da_xN, obs_partial, da_Lag)