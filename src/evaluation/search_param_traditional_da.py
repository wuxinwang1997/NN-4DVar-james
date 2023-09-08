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
from src.utils.utils import str2bool
import xarray as xr
import torch
from src.da_methods.fdvar_l96 import My4DVar
from src.models.components.lorenz96 import Lorenz96_torch as L96torch
from src.models.components.lorenz96 import Lorenz96_partial_Id_Obs
from src.models.components.qg import torch_qg, LP_setup
from src.utils.utils import ens_compatible, simulate
np.random.seed(2023)

def search_param(data_dir, system, years, N_trials, dim, da_method, da_N, da_Inf, da_Bx, da_rot, da_loc_rad, da_xN, obs_partial, da_Lag=0):
    if system == 'Lorenz96':
        spin_up = int(1000 * 0.05)
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
        # max_offset = jj[-1] - jj[0]
        # def random_offset(jj, t):
        #     rstream = np.random.RandomState()
        #     rstream.seed(2023)
        #     u = rstream.rand()
        #     return int(np.floor(max_offset * u))

        def obs_inds(t):
            return jj #+ random_offset(t)

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

        # numpy to save x^b, x^a and x^t
        # numpy to save observations
        obs = np.zeros(shape=(tseq.Ko, tseq.dko, Nx), dtype=np.float32)
        # C = np.random.normal(loc=0, scale=0.001, size=N_trials)

        X0 = modelling.GaussRV(mu=x0, C=0.001)
        HMM = modelling.HiddenMarkovModel(Dyn, Obs, tseq, X0)
        HMM.X0 = modelling.GaussRV(mu=x0, C=0.001)
        HMM.liveplotters = Lorenz96.LPs(jj)

    elif system == 'QG':
        qg = torch_qg()
        Nx = np.prod(qg.shape)
        Dyn = {
            'M': Nx,
            'model': qg.forward,
            'noise': 0,
            'name': 'QG',
        }

        OneYear = qg.dtout * 24 * 365
        # Considering that I have 8GB mem on the Mac, and the estimate:
        # ≈ (8 bytes/float)*(129² float/stat)*(7 stat/k) * K,
        # it should be possible to run experiments of length (K) < 8000.
        t = modelling.Chronology(dt=qg.dtout, dko=4, T=years * OneYear, BurnIn=250)
        # In my opinion the burn in should be 400.
        # Sakov also used 10 repetitions.

        X0 = modelling.RV(M=Dyn['M'], file=f'/home/wangwx/dpr_data/samples/QG_MREFIN{qg.MREFIN}_scheme{qg.scheme}_samples.npz')

        ############################
        # Observation settings
        ############################

        # This will look like satellite tracks when plotted in 2D
        Ny = 300
        # Ny = 200
        jj = modelling.linspace_int(Dyn['M'], Ny)

        # Want: random_offset(t1)==random_offset(t2) if t1==t2.
        # Solutions: (1) use caching (ensure maxsize=inf) or (2) stream seeding.
        # Either way, use a local random stream to avoid interfering with global stream
        # (and e.g. ensure equal outcomes for 1st and 2nd run of the python session).
        rstream = np.random.RandomState()
        max_offset = jj[1]-jj[0]

        def random_offset(t):
            rstream.seed(int(t/qg.dtout*100))
            u = rstream.rand()
            return int(np.floor(max_offset * u))

        def obs_inds(t):
            return jj + random_offset(t)

        # @modelling.ens_compatible
        @ens_compatible
        def hmod(E, t):
            return E[obs_inds(t)]

        # Localization.
        batch_shape = [2, 2]  # width (in grid points) of each state batch.
        # Increasing the width
        #  => quicker analysis (but less rel. speed-up by parallelzt., depending on NPROC)
        #  => worse (increased) rmse (but width 4 is only slightly worse than 1);
        #     if inflation is applied locally, then rmse might actually improve.
        localizer = nd_Id_localization(qg.shape[::-1], batch_shape[::-1], obs_inds, periodic=False)

        Obs = {
            'M': Ny,
            'model': hmod,
            'noise': modelling.GaussRV(C=4*np.eye(Ny)),
            'localizer': localizer,
        }

        # Moving localization mask for smoothers:
        Obs['loc_shift'] = lambda ii, dt: ii  # no movement (suboptimal, but easy)

        # Jacobian left unspecified coz it's (usually) employed by methods that
        # compute full cov, which in this case is too big.
        ############################
        # Other
        ############################
        HMM = modelling.HiddenMarkovModel(Dyn, Obs, t, X0, LP=LP_setup(obs_inds))

    # data assimilation method to be used to generate the dataset
    if da_method == 'OI':
        xp = da.OptInterp()
    if da_method == 'Var3D':
        xp = da.Var3D(xB=da_Bx)
    if da_method == 'Var4D':
        xp = da.Var4D(B='clim', Lag=1, nIter=5, xB=da_Bx, wtol=1e-7)
    if da_method == 'My4DVar':
        # xp = My4DVar(B='clim', Lag=1, nIter=1, xB=da_Bx, lr=5e-1, max_iter=5, tolerance_grad=1e-9, tolerance_change=1e-12, history_size=50)
        xp = My4DVar(B='clim', Lag=1, nIter=2, xB=da_Bx, lr=5e-1, max_iter=5, tolerance_grad=1e-9, tolerance_change=1e-12, history_size=50)
    if da_method == 'DEnKF':
        xp = da.EnKF('DEnKF',N=da_N, infl=da_Inf, rot=da_rot)
    if da_method == 'EnKF':
        xp = da.EnKF('Sqrt',N=da_N, infl=da_Inf, rot=da_rot)
    if da_method == 'LETKF':
        xp = da.LETKF(N=da_N , infl=da_Inf, rot=da_rot, loc_rad=da_loc_rad)
    if da_method == 'iEnKS':
        xp = da.iEnKS('Sqrt', N=da_N, Lag=da_Lag, infl=da_Inf, rot=da_rot, bundle=True)

    xp.seed = 2023
    my_HMM = HMM.copy()
    xx, yy, yy_ = simulate(my_HMM)

    # 需要实验往前一个同化窗口的观测的获取，观测单独保存数据
    if da_method == 'My4DVar':
        HMM.tseq.Ko -= 1
        xp.assimilate(HMM, xx[:-HMM.tseq.dko], yy, yy_[HMM.tseq.dko:])
    else:
        HMM.tseq.Ko -= 1
        xp.assimilate(HMM, xx[:-HMM.tseq.dko], yy[:-1], yy_[HMM.tseq.dko:])

    xa = xp.stats.mu.a
    xb = xp.stats.mu.f

    xb = xb[~np.isnan(xb).any(axis=1), :][int(spin_up / HMM.tseq.dto) - 1:]
    xa = xa[~np.isnan(xa).any(axis=1), :][int(spin_up / HMM.tseq.dto) - 1:]
    xx = xx[int(spin_up / HMM.tseq.dt):]

    xx = xr.DataArray(
        xx,
        dims=['lead_time', 'grid'],
        coords={
            'lead_time': np.arange(0, years*OneYear+HMM.tseq.dt, HMM.tseq.dt).astype(np.float32),
            'grid': np.arange(Nx),
        },
        name='x'
    )

    xb = xr.DataArray(
        xb,
        dims=['lead_time', 'grid'],
        coords={
            'lead_time': np.arange(0, years*OneYear, HMM.tseq.dto).astype(np.float32),
            'grid': np.arange(Nx), 
        },
        name='x'
    )

    rmse_xb = xr.DataArray(
        np.sqrt(np.mean((xb-xx.sel(lead_time=xb['lead_time'].values))**2, axis=-1)),
        dims=['lead_time'],
        coords={
            'lead_time': np.arange(0, years*OneYear, HMM.tseq.dto).astype(np.float32),
        },
        name='x'
    )

    xa = xr.DataArray(
        xa,
        dims=['lead_time', 'grid'],
        coords={
            'lead_time': np.arange(0, years*OneYear, HMM.tseq.dto).astype(np.float32),
            'grid': np.arange(Nx), 
        },
        name='x'
    )

    rmse_xa = xr.DataArray(
        np.sqrt(np.mean((xa-xx.sel(lead_time=xa['lead_time']))**2, axis=-1)),
        dims=['lead_time'],
        coords={
            'lead_time': np.arange(0, years*OneYear, HMM.tseq.dto).astype(np.float32),
        },
        name='x'
    )

    obs = xr.DataArray(
        yy_[int(spin_up/HMM.tseq.dt):],
        dims=['lead_time', 'grid'],
        coords={
            'lead_time': np.arange(0, years*OneYear+HMM.tseq.dt, HMM.tseq.dt).astype(np.float32),
            'grid': jj, 
        },
        name='x'
    )

    if not os.path.exists(data_dir):
        os.mkdir(data_dir)

    print(rmse_xb.mean(), rmse_xa.mean())

    xx.to_netcdf(f'{data_dir}/{system}_gt_{da_method}_T{years}_N_trails{N_trials}_N{dim}_daN{da_N}_daInf{da_Inf}_daBx{da_Bx}_darot{da_rot}_dalocrad{da_loc_rad}_obspartial{obs_partial}_daLag{da_Lag}.nc')
    xb.to_netcdf(f'{data_dir}/{system}_xb_{da_method}_T{years}_N_trails{N_trials}_N{dim}_daN{da_N}_daInf{da_Inf}_daBx{da_Bx}_darot{da_rot}_dalocrad{da_loc_rad}_obspartial{obs_partial}_daLag{da_Lag}.nc')
    rmse_xb.to_netcdf(f'{data_dir}/{system}_rmse_xb_{da_method}_T{years}_N_trails{N_trials}_N{dim}_daN{da_N}_daInf{da_Inf}_daBx{da_Bx}_darot{da_rot}_dalocrad{da_loc_rad}_obspartial{obs_partial}_daLag{da_Lag}.nc')
    xa.to_netcdf(f'{data_dir}/{system}_xa_{da_method}_T{years}_N_trails{N_trials}_N{dim}_daN{da_N}_daInf{da_Inf}_daBx{da_Bx}_darot{da_rot}_dalocrad{da_loc_rad}_obspartial{obs_partial}_daLag{da_Lag}.nc')
    rmse_xa.to_netcdf(f'{data_dir}/{system}_rmse_xa_{da_method}_T{years}_N_trails{N_trials}_N{dim}_daN{da_N}_daInf{da_Inf}_daBx{da_Bx}_darot{da_rot}_dalocrad{da_loc_rad}_obspartial{obs_partial}_daLag{da_Lag}.nc')
    obs.to_netcdf(f'{data_dir}/{system}_obs_{da_method}_T{years}_N_trails{N_trials}_N{dim}_daN{da_N}_daInf{da_Inf}_daBx{da_Bx}_darot{da_rot}_dalocrad{da_loc_rad}_obspartial{obs_partial}_daLag{da_Lag}.nc')

    #return


def prepare_parser():
    parser = argparse.ArgumentParser(description='Generate Background')
    parser.add_argument(
        '--data_dir',
        type=str,
        help='path to save the dataset',
        default='./data/search_param'
    )

    parser.add_argument(
        '--system',
        type=str,
        help='name of the experimenting system',
        default='Lorenz96'
    )

    parser.add_argument(
        '--years',
        type=float,
        help = 'years for experiments',
        default=1
    )

    parser.add_argument(
        '--N_trials',
        type = int,
        help = 'trials for data generation',
        default=1
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
        default='EnKF'
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
    system = args.system
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
    print(da_Inf, da_rot)
    search_param(data_dir, system, years, N_trials, dim, da_method, da_N, da_Inf, da_Bx, da_rot, da_loc_rad, da_xN, obs_partial, da_Lag)