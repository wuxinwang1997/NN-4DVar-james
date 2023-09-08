import torch
import torch.nn as nn
import sys
sys.path.append('.')
import numpy as np
import os
from src.da_methods.fdvar_l96 import My4DVar
from src.models.components.lorenz96 import Lorenz96_torch as L96torch
from src.models.components.lorenz96 import Lorenz96_partial_Id_Obs
from src.models.components.qg import torch_qg, LP_setup
import dapper.mods as modelling
import dapper.mods.Lorenz96 as Lorenz96
from dapper.tools.localization import nd_Id_localization
import dapper.da_methods as da
from dapper.da_methods import da_method
from dapper.tools.progressbar import progbar
from pathlib import Path
import time
from src.models.nn4dvar_module import NN4DVarLitModule
from src.models.fdvarnet_module import FDVarNNLitModule
from src.utils.utils import simulate
import argparse
import xarray as xr
from src.utils.utils import str2bool
np.random.seed(2023)

@da_method()
class NN4DVar:
    """Optimal Interpolation -- a baseline/reference method.

    Uses the Kalman filter equations,
    but with a prior from the Climatology.
    """
    model: nn.Module = None
    mean: dict = None
    std: dict = None
    def assimilate(self, HMM, xx, yy, yy_all):
        P  = HMM.X0.C.full
        mu = np.squeeze(HMM.X0.sample(1), axis=0)
        self.stats.assess(0, mu=mu, Cov=P)
        for k, ko, t, dt in progbar(HMM.tseq.ticker):
            # Forecast
            mu = HMM.Dyn(mu, t-dt, dt)
            if isinstance(mu, torch.Tensor):
                mu = np.squeeze(mu.detach().numpy(), axis=0)
            if ko is not None:
                self.stats.assess(k, ko, 'f', mu=mu, Cov=P)
                # Analysis
                xb = torch.from_numpy(np.expand_dims(np.expand_dims((mu-self.mean['xb'])/self.std['xb'], axis=0), axis=0)).to('cpu', dtype=torch.float32)
                pred_xb = np.expand_dims(np.ones_like(yy_all[ko]), axis=0)
                pred_xb[:,0,:] = xb[:,0,:]
                for i in range(HMM.tseq.dko-1):
                    pred_xb[:,i+1:i+2,:] = (HMM.Dyn(pred_xb[:,i:i+1,:]*self.std['xb']+self.mean['xb'], t+i*dt, dt)-self.mean['xb'])/self.std['xb']
                obs_all = torch.from_numpy(np.expand_dims((yy_all[ko]-self.mean['obs'])/self.std['obs'], axis=0)).to('cpu', dtype=torch.float32)
                # in_obs_idx = np.zeros_like(obs_all.detach().cpu().numpy())
                # in_obs_idx[:,:,np.asarray(np.where(obs_all != 0))[-1].reshape((HMM.tseq.dko, -1))[0,:]] = 1
                # in_obs_idx = torch.from_numpy(in_obs_idx).to('cpu', dtype=torch.float32)
                # obs_all = obs_all * in_obs_idx + (1-in_obs_idx) * torch.from_numpy(pred_xb).to('cpu', dtype=torch.float32)
                obs_all = torch.where(obs_all!=0, obs_all, torch.from_numpy(pred_xb).to('cpu', dtype=torch.float32))
                mu = self.model(xb, obs_all)
                mu = np.squeeze(np.squeeze(mu.detach().numpy(), axis=0), axis=0) * self.std['xa'] + self.mean['xa']
            self.stats.assess(k, ko, mu=mu, Cov=P)

@da_method()
class FDVarNet:
    """Optimal Interpolation -- a baseline/reference method.

    Uses the Kalman filter equations,
    but with a prior from the Climatology.
    """
    model: nn.Module = None
    mean: dict = None
    std: dict = None
    obs_idx: np.ndarray = None
    def assimilate(self, HMM, xx, yy, yy_all):
        P  = HMM.X0.C.full
        mu = np.squeeze(HMM.X0.sample(1), axis=0)
        self.stats.assess(0, mu=mu, Cov=P)
        for k, ko, t, dt in progbar(HMM.tseq.ticker):
            # Forecast
            mu = HMM.Dyn(mu, t-dt, dt)
            if isinstance(mu, torch.Tensor):
                mu = np.squeeze(mu.detach().numpy(), axis=0)
            if ko is not None:
                self.stats.assess(k, ko, 'f', mu=mu, Cov=P)
                # Analysis
                with torch.set_grad_enabled(True):
                    xb = torch.from_numpy(np.expand_dims(np.expand_dims(np.expand_dims(mu, axis=0), axis=0), axis=-1))
                    xb_input = torch.autograd.Variable(xb, requires_grad=True).to('cpu', dtype=torch.float32)
                    obs_all = np.zeros((1, 1, HMM.tseq.dko, HMM.Dyn.M)).astype(np.float32)
                    obs_all[:,:] = yy_all[ko]
                    obs_all = (torch.from_numpy(obs_all).permute(0, 1, 3, 2)).to('cpu', dtype=torch.float32)
                    mu, hidden_new, cell_new, normgrad = self.model(xb_input, obs_all, self.obs_idx, None, None)
                mu = np.squeeze(np.squeeze(np.squeeze(mu.detach().numpy(), axis=0), axis=0), axis=-1)
            self.stats.assess(k, ko, mu=mu, Cov=P)

def eval_cycles(res_dir, pretrain_dir, system, years, mlda_name, da_model, pred_len, N_trials, dim, da_method, da_N, da_Inf, da_Bx, da_rot, da_loc_rad, da_xN, obs_partial, normalize, obserr=1.0, da_Lag=0):
    if normalize:
        stats = np.load(f'data/train/L96_stats_{da_method}_T4_N_trials2_N{dim}_daN{da_N}_daInf{da_Inf}_daBx{da_Bx}_darot{da_rot}_dalocrad{da_loc_rad}_obspartial{obs_partial}_daLag{da_Lag}.npz')
        mean = {'xb': stats['mean_xb'], 'xa':stats['mean_xa'], 'obs':stats['mean_obs']}
        std = {'xb': stats['std_xb'], 'xa': stats['std_xa'], 'obs': stats['std_obs']}
    else:
        mean = {'xb': 0, 'xa': 0, 'obs': 0}
        std = {'xb': 1, 'xa': 1, 'obs': 1}

    da_model_ckpt = f'{pretrain_dir}/L{dim}/obs{obs_partial}/{da_method.lower()}/predlen{int(pred_len)}/{da_model}/checkpoints/{da_model}.ckpt'
    # da_model_ckpt = '/mnt/d/Study/Lab/work/idea1/NN4DVar/logs/train/runs/2023-05-28_19-05-50/checkpoints/epoch_049.ckpt'
    da_model_name = da_model

    spin_up = int(2000 * 0.05)
    OneYear = 0.05 * (24 / 6) * 365
    # Sakov uses K=300000, BurnIn=1000*0.05
    tseq = modelling.Chronology(0.01, dto=0.05, T=spin_up + years * OneYear, Tplot=Lorenz96.Tplot,
                                BurnIn=2 * Lorenz96.Tplot)

    Nx = dim
    x0 = Lorenz96.x0(Nx)

    # select random observation grids
    # jj = np.zeros(shape=(Nx))
    # jj[:int(obs_partial * Nx)] = int(1)
    # np.random.shuffle(jj)
    # jj = np.asarray(np.where(jj == 1)[0])
    if obs_partial == 0.75:
        jj = np.ones(shape=(Nx))
        jj[np.asarray(np.arange(1, Nx, 4))] = 0
        jj = np.asarray(np.where(jj==1)[0])
    else:
        jj = np.asarray(np.arange(Nx))

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
        Obs['noise'] = obserr
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
        Obs['noise'] = obserr
        Obs['localizer'] = nd_Id_localization((Nx,), (2,), jj)

    xxs, xb_mls, xa_mls, rmse_xb_mls, rmse_xa_mls, xb_tradition, xa_tradition, rmse_xb_traditions, rmse_xa_traditions, yys, yy_all = [], [], [], [], [], [], [], [], [], [], []
    for i in range(10):

        X0 = modelling.GaussRV(mu=x0, C=0.001)
        HMM = modelling.HiddenMarkovModel(Dyn, Obs, tseq, X0)
        HMM.liveplotters = Lorenz96.LPs(jj)
        xx, yy, yy_ = simulate(HMM)

        # HMM.tseq.Ko = HMM.tseq.Ko - 1

        # # obs_all = yy_[HMM.tseq.dko:-1].reshape(-1, HMM.tseq.dko, dim)
        # obs_all = np.ones((yy.shape[0]-1, HMM.tseq.dko, dim)) * mean['obs']
        # obs_all[:,:,jj] = yy_[HMM.tseq.dko:-1].reshape(-1, HMM.tseq.dko, int(obs_partial*dim))

        # if mlda_name == 'NN4DVar':
        #     da_module = NN4DVarLitModule.load_from_checkpoint(da_model_ckpt)
        #     da_model = da_module.net.to(device='cpu').eval()
        #     xp = NN4DVar(model=da_model, mean=mean, std=std)
        # elif mlda_name == '4DVarNet':
        #     da_module = FDVarNNLitModule.load_from_checkpoint(da_model_ckpt)
        #     da_model = da_module.model.to(device='cpu').eval()
        #     obs_idx = np.ones(shape=(Nx))
        #     if obs_partial == 0.75:
        #         obs_idx[np.asarray(np.arange(1, Nx, 4))] = 0
        #     xp = FDVarNet(model=da_model, mean=mean, std=std, obs_idx=torch.from_numpy(np.expand_dims(np.expand_dims(np.expand_dims(obs_idx, axis=-1), axis=0), axis=0)))

        # xp.seed = 2023

        # xp.assimilate(HMM, xx[:-HMM.tseq.dko], yy[:-1], obs_all)

        # xb = xp.stats.mu.f
        # xa = xp.stats.mu.a

        # xb = xb[~np.isnan(xb).any(axis=1), :][int(spin_up / HMM.tseq.dto):]
        # xa = xa[~np.isnan(xa).any(axis=1), :][int(spin_up / HMM.tseq.dto):]
        # xx_ = xx[HMM.tseq.dko:-HMM.tseq.dko:HMM.tseq.dko][int(spin_up / HMM.tseq.dto):]

        # rmse_xb = np.sqrt(np.mean((xb - xx_) ** 2, axis=-1))
        # rmse_xa = np.sqrt(np.mean((xa - xx_) ** 2, axis=-1))
        # print('ML Xa RMSE: ', np.mean(rmse_xa))
        # print('ML Xb RMSE: ', np.mean(rmse_xb))
        # # xxs.append(xx_)
        # # yys.append(yy[int(spin_up / HMM.tseq.dto):-1])
        # xb_mls.append(xb)
        # xa_mls.append(xa)
        # rmse_xb_mls.append(rmse_xb)
        # rmse_xa_mls.append(rmse_xa)

        # # yy = yy[int(spin_up / HMM.tseq.dto) - 1:-1]
        # # print('Obs RMSE: ', np.mean((yy[int(spin_up / HMM.tseq.dto) - 1:-1] - xx[int(spin_up / HMM.tseq.dt):-HMM.tseq.dko:HMM.tseq.dko]) ** 2))
        # yy_all.append(yy_[int(spin_up / HMM.tseq.dt): -1])
        # HMM.tseq.Ko = HMM.tseq.Ko + 1

        # data assimilation method to be used to generate the dataset
        if da_method == 'OI':
            xp = da.OptInterp()
        if da_method == 'Var3D':
            xp = da.Var3D(xB=da_Bx)
        if da_method == 'Var4D':
            xp = da.Var4D(B='clim', Lag=1, nIter=12, xB=da_Bx, wtol=1e-5)
        if da_method == 'My4DVar':
            xp = My4DVar(B='clim', Lag=1, nIter=2, xB=da_Bx, lr=5e-1, max_iter=5, tolerance_grad=1e-7, tolerance_change=1e-9,
                         history_size=50)
        if da_method == 'DEnKF':
            xp = da.EnKF('DEnKF', N=da_N, infl=da_Inf, rot=da_rot)
        if da_method == 'EnKF':
            xp = da.EnKF('Sqrt', N=da_N, infl=da_Inf, rot=da_rot)
        if da_method == 'LETKF':
            xp = da.LETKF(N=da_N, infl=da_Inf, rot=da_rot, loc_rad=da_loc_rad)
        if da_method == 'iEnKS':
            xp = da.iEnKS('Sqrt', N=da_N, Lag=da_Lag, infl=da_Inf, rot=da_rot, bundle=True)  # xN=da_xN)
        xp.seed = 2023
        if da_method == 'My4DVar':
            HMM.tseq.Ko -= 1
            xp.assimilate(HMM, xx[:-HMM.tseq.dko], yy[:-1], yy_[HMM.tseq.dko:])
        else:
            HMM.tseq.Ko -= 1
            xp.assimilate(HMM, xx[:-HMM.tseq.dko], yy[:-1], yy_[HMM.tseq.dko:])
        HMM.tseq.Ko = HMM.tseq.Ko + 1
        xa = xp.stats.mu.a
        xb = xp.stats.mu.f
        
        xb = xb[~np.isnan(xb).any(axis=1), :][int(spin_up / HMM.tseq.dto):]
        xa = xa[~np.isnan(xa).any(axis=1), :][int(spin_up / HMM.tseq.dto):]
        xx = xx[HMM.tseq.dko:-HMM.tseq.dko:HMM.tseq.dko][int(spin_up / HMM.tseq.dto):]
        yy = yy[int(spin_up / HMM.tseq.dto):-1]
        yy_ = yy_[HMM.tseq.dko:-1][int(spin_up / HMM.tseq.dt):]

        # rmse_xb = np.sqrt(np.mean((xb-xx[(np.arange(0, years*OneYear, HMM.tseq.dto)/HMM.tseq.dt).astype(np.int32)])**2, axis=-1))
        # rmse_xa = np.sqrt(np.mean((xa-xx[(np.arange(0, years*OneYear, HMM.tseq.dto)/HMM.tseq.dt).astype(np.int32)])**2, axis=-1))
        rmse_xb = np.sqrt(np.mean((xb - xx) ** 2, axis=-1))
        rmse_xa = np.sqrt(np.mean((xa - xx) ** 2, axis=-1))
        # print(f'{da_method} Xa RMSE: ', rmse_xa[:100])
        # print(f'{da_method} Xb RMSE: ', rmse_xb[:100])
        print(f'{da_method} Xa RMSE: ', np.mean(rmse_xa))
        print(f'{da_method} Xb RMSE: ', np.mean(rmse_xb))
        # print(f'Obs RMSE: ', np.mean((yy - xx) ** 2))
        xb_tradition.append(xb)
        xa_tradition.append(xa)
        rmse_xa_traditions.append(rmse_xa)
        rmse_xb_traditions.append(rmse_xb)
        
        xxs.append(xx)
        yys.append(yy)
        yy_all.append(yy_)

    # xx = xr.DataArray(
    #     np.stack(xxs, axis=0),
    #     dims=['trail', 'lead_time', 'grid'],
    #     coords={
    #         # 'lead_time': np.arange(0, years * OneYear + HMM.tseq.dt, HMM.tseq.dt).astype(np.float32),
    #         'trail': np.arange(10),
    #         'lead_time': np.arange(0, years * OneYear-HMM.tseq.dto, HMM.tseq.dto).astype(np.float32),
    #         'grid': np.arange(Nx),
    #     },
    #     name='x'
    # )

    # xb_ml = xr.DataArray(
    #     np.asarray(xb_mls),
    #     dims=['trail', 'lead_time', 'grid'],
    #     coords={
    #         'trail': np.arange(10),
    #         'lead_time': np.arange(0, years * OneYear-HMM.tseq.dto, HMM.tseq.dto).astype(np.float32),
    #         'grid': np.arange(Nx),
    #     },
    #     name='x'
    # )

    # rmse_xb_ml = xr.DataArray(
    #     # np.sqrt(np.mean((xb - xx.sel(lead_time=xb['lead_time'].values)) ** 2, axis=-1)),
    #     np.asarray(rmse_xb_mls),
    #     dims=['trail', 'lead_time'],
    #     coords={
    #         'trail': np.arange(10),
    #         'lead_time': np.arange(0, years * OneYear-HMM.tseq.dto, HMM.tseq.dto).astype(np.float32),
    #     },
    #     name='x'
    # )

    # xa_ml = xr.DataArray(
    #     np.asarray(xa_mls),
    #     dims=['trail', 'lead_time', 'grid'],
    #     coords={
    #         'trail': np.arange(10),
    #         'lead_time': np.arange(0, years * OneYear-HMM.tseq.dto, HMM.tseq.dto).astype(np.float32),
    #         'grid': np.arange(Nx),
    #     },
    #     name='x'
    # )

    # rmse_xa_ml = xr.DataArray(
    #     # np.sqrt(np.mean((xa - xx.sel(lead_time=xa['lead_time'])) ** 2, axis=-1)),
    #     np.asarray(rmse_xa_mls),
    #     dims=['trail', 'lead_time'],
    #     coords={
    #         'trail': np.arange(10),
    #         'lead_time': np.arange(0, years * OneYear-HMM.tseq.dto, HMM.tseq.dto).astype(np.float32),
    #     },
    #     name='x'
    # )

    xb_tradition = xr.DataArray(
        np.asarray(xb_tradition),
        dims=['trail', 'lead_time', 'grid'],
        coords={
            'trail': np.arange(10),
            'lead_time': np.arange(0, years * OneYear-HMM.tseq.dto, HMM.tseq.dto).astype(np.float32),
            'grid': np.arange(Nx),
        },
        name='x'
    )
    
    rmse_xb_tradition = xr.DataArray(
        # np.sqrt(np.mean((xb - xx.sel(lead_time=xb['lead_time'].values)) ** 2, axis=-1)),
        np.asarray(rmse_xb_traditions),
        dims=['trail', 'lead_time'],
        coords={
            'trail': np.arange(10),
            'lead_time': np.arange(0, years * OneYear-HMM.tseq.dto, HMM.tseq.dto).astype(np.float32),
        },
        name='x'
    )
    
    xa_tradition = xr.DataArray(
        np.asarray(xa_tradition),
        dims=['trail', 'lead_time', 'grid'],
        coords={
            'trail': np.arange(10),
            'lead_time': np.arange(0, years * OneYear-HMM.tseq.dto, HMM.tseq.dto).astype(np.float32),
            'grid': np.arange(Nx),
        },
        name='x'
    )
    
    rmse_xa_tradition = xr.DataArray(
        # np.sqrt(np.mean((xa - xx.sel(lead_time=xa['lead_time'])) ** 2, axis=-1)),
        np.asarray(rmse_xa_traditions),
        dims=['trail', 'lead_time'],
        coords={
            'trail': np.arange(10),
            'lead_time': np.arange(0, years * OneYear-HMM.tseq.dto, HMM.tseq.dto).astype(np.float32),
        },
        name='x'
    )

    # obs = xr.DataArray(
    #     np.asarray(yys),
    #     dims=['trail', 'lead_time', 'grid'],
    #     coords={
    #         # 'trial': np.arange(0, N_trials, 1),
    #         'trail': np.arange(10),
    #         'lead_time': np.arange(0, years * OneYear-HMM.tseq.dto, HMM.tseq.dto).astype(np.float32),
    #         'grid': jj,
    #     },
    #     name='x'
    # )

    # obs_all = xr.DataArray(
    #     np.asarray(yy_all),
    #     dims=['trail', 'lead_time', 'grid'],
    #     coords={
    #         # 'trial': np.arange(0, N_trials, 1),
    #         'trail': np.arange(10),
    #         'lead_time': np.arange(0, yy_all[0].shape[0] * HMM.tseq.dt, HMM.tseq.dt).astype(np.float32),
    #         'grid': jj,
    #     },
    #     name='x'
    # )

    rmse_xa_ml_mean = rmse_xa_ml.mean(dim='lead_time')
    rmse_xb_ml_mean = rmse_xb_ml.mean(dim='lead_time')
    print('ML Xa RMSE mean: ', np.mean(rmse_xa_ml_mean), ' std: ', np.std(rmse_xa_ml_mean))
    print('ML Xb RMSE mean: ', np.mean(rmse_xb_ml_mean), ' std: ', np.std(rmse_xb_ml_mean))
    
    # rmse_xa_trad_mean = rmse_xa_tradition.mean(dim='lead_time')
    # rmse_xb_trad_mean = rmse_xb_tradition.mean(dim='lead_time')
    # print(f'{da_method} Xa RMSE mean: ', np.mean(rmse_xa_trad_mean), ' std: ', np.std(rmse_xa_trad_mean))
    # print(f'{da_method} Xa RMSE mean: ', np.mean(rmse_xb_trad_mean), ' std: ', np.std(rmse_xb_trad_mean))

    if not os.path.exists(f'{res_dir}/eval_cycles/obspartial{obs_partial}/{da_method.lower()}/pred_len{int(pred_len)}/{da_model_name}'):
        os.mkdir(f'{res_dir}/eval_cycles/obspartial{obs_partial}/{da_method.lower()}/pred_len{int(pred_len)}/{da_model_name}')

    # xx.to_netcdf(
    #     f'{res_dir}/eval_cycles/obspartial{obs_partial}/{da_method.lower()}/pred_len{int(pred_len)}/{da_model_name}/{system}_gt_{da_method}_T{years}_N_trails{N_trials}_N{dim}_daN{da_N}_daInf{da_Inf}_daBx{da_Bx}_darot{da_rot}_dalocrad{da_loc_rad}_obspartial{obs_partial}_daLag{da_Lag}.nc')
    # xb_ml.to_netcdf(
    #     f'{res_dir}/eval_cycles/obspartial{obs_partial}/{da_method.lower()}/pred_len{int(pred_len)}/{da_model_name}/{system}_xb_ml_{da_method}_T{years}_N_trails{N_trials}_N{dim}_daN{da_N}_daInf{da_Inf}_daBx{da_Bx}_darot{da_rot}_dalocrad{da_loc_rad}_obspartial{obs_partial}_daLag{da_Lag}.nc')
    # rmse_xb_ml.to_netcdf(
    #     f'{res_dir}/eval_cycles/obspartial{obs_partial}/{da_method.lower()}/pred_len{int(pred_len)}/{da_model_name}/{system}_rmse_xb_ml_{da_method}_T{years}_N_trails{N_trials}_N{dim}_daN{da_N}_daInf{da_Inf}_daBx{da_Bx}_darot{da_rot}_dalocrad{da_loc_rad}_obspartial{obs_partial}_daLag{da_Lag}.nc')
    # xa_ml.to_netcdf(
    #     f'{res_dir}/eval_cycles/obspartial{obs_partial}/{da_method.lower()}/pred_len{int(pred_len)}/{da_model_name}/{system}_xa_ml_{da_method}_T{years}_N_trails{N_trials}_N{dim}_daN{da_N}_daInf{da_Inf}_daBx{da_Bx}_darot{da_rot}_dalocrad{da_loc_rad}_obspartial{obs_partial}_daLag{da_Lag}.nc')
    # rmse_xa_ml.to_netcdf(
    #     f'{res_dir}/eval_cycles/obspartial{obs_partial}/{da_method.lower()}/pred_len{int(pred_len)}/{da_model_name}/{system}_rmse_xa_ml_{da_method}_T{years}_N_trails{N_trials}_N{dim}_daN{da_N}_daInf{da_Inf}_daBx{da_Bx}_darot{da_rot}_dalocrad{da_loc_rad}_obspartial{obs_partial}_daLag{da_Lag}.nc')
    # xb_tradition.to_netcdf(
    #     f'{res_dir}/eval_cycles/obspartial{obs_partial}/{da_method.lower()}/pred_len{int(pred_len)}/{da_model_name}/{system}_xb_tradition_{da_method}_T{years}_N_trails{N_trials}_N{dim}_daN{da_N}_daInf{da_Inf}_daBx{da_Bx}_darot{da_rot}_dalocrad{da_loc_rad}_obspartial{obs_partial}_daLag{da_Lag}.nc')
    # rmse_xb_tradition.to_netcdf(
    #     f'{res_dir}/eval_cycles/obspartial{obs_partial}/{da_method.lower()}/pred_len{int(pred_len)}/{da_model_name}/{system}_rmse_xb_tradition_{da_method}_T{years}_N_trails{N_trials}_N{dim}_daN{da_N}_daInf{da_Inf}_daBx{da_Bx}_darot{da_rot}_dalocrad{da_loc_rad}_obspartial{obs_partial}_daLag{da_Lag}.nc')
    # xa_tradition.to_netcdf(
    #     f'{res_dir}/eval_cycles/obspartial{obs_partial}/{da_method.lower()}/pred_len{int(pred_len)}/{da_model_name}/{system}_xa_tradition_{da_method}_T{years}_N_trails{N_trials}_N{dim}_daN{da_N}_daInf{da_Inf}_daBx{da_Bx}_darot{da_rot}_dalocrad{da_loc_rad}_obspartial{obs_partial}_daLag{da_Lag}.nc')
    # rmse_xa_tradition.to_netcdf(
    #     f'{res_dir}/eval_cycles/obspartial{obs_partial}/{da_method.lower()}/pred_len{int(pred_len)}/{da_model_name}/{system}_rmse_xa_tradition_{da_method}_T{years}_N_trails{N_trials}_N{dim}_daN{da_N}_daInf{da_Inf}_daBx{da_Bx}_darot{da_rot}_dalocrad{da_loc_rad}_obspartial{obs_partial}_daLag{da_Lag}.nc')
    # obs.to_netcdf(
    #     f'{res_dir}/eval_cycles/obspartial{obs_partial}/{da_method.lower()}/pred_len{int(pred_len)}/{da_model_name}/{system}_obs_{da_method}_T{years}_N_trails{N_trials}_N{dim}_daN{da_N}_daInf{da_Inf}_daBx{da_Bx}_darot{da_rot}_dalocrad{da_loc_rad}_obspartial{obs_partial}_daLag{da_Lag}.nc')
    # obs_all.to_netcdf(
    #     f'{res_dir}/eval_cycles/obspartial{obs_partial}/{da_method.lower()}/pred_len{int(pred_len)}/{da_model_name}/{system}_obs_all_{da_method}_T{years}_N_trails{N_trials}_N{dim}_daN{da_N}_daInf{da_Inf}_daBx{da_Bx}_darot{da_rot}_dalocrad{da_loc_rad}_obspartial{obs_partial}_daLag{da_Lag}.nc')

def prepare_parser():
    parser = argparse.ArgumentParser(description='Generate Background')
    parser.add_argument(
        '--res_dir',
        type=str,
        help='path to save the dataset',
        default='./data/evaluate'
    )

    parser.add_argument(
        '--pretrain_dir',
        type=str,
        help='path to save the dataset',
        default='./pretrain'
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
        help='years for experiments',
        default=4
    )

    parser.add_argument(
        '--mlda_name',
        type=str,
        help='ml damodel name',
        default='NN4DVar'
    )

    parser.add_argument(
        '--da_model',
        type=str,
        help='model to do assimilation',
        default='tinynet'
    )

    parser.add_argument(
        '--pred_len',
        type=float,
        help='prediction length to train the da model',
        default=0
    )

    parser.add_argument(
        '--N_trials',
        type=int,
        help='trials for data generation',
        default=2
    )

    parser.add_argument(
        '--dim',
        type=int,
        help='dimension for Lorenz96 system',
        default=40
    )

    parser.add_argument(
        '--da_method',
        type=str,
        help='data assimilation method utilized to generate dataset',
        default='EnKF'
    )

    parser.add_argument(
        '--da_N',
        type=int,
        help='ensemble size for EnDA',
        default=20
    )

    parser.add_argument(
        '--da_Inf',
        type=float,
        help='Inflation hyperparameter for EnDA',
        default=1.0
    )

    parser.add_argument(
        '--da_Bx',
        type=float,
        help='Covariance multiple ratio',
        default=1.0
    )

    parser.add_argument(
        '--da_rot',
        type=str2bool,
        help='Rotation hyperparameter for EnDA',
        default=False
    )

    parser.add_argument(
        '--da_loc_rad',
        type=float,
        help='Localization ratio hyperparameter for EnDA',
        default=1.0
    )

    parser.add_argument(
        '--da_xN',
        type=float,
        help='xN hyperparameter for EnDA',
        default=1.0
    )

    parser.add_argument(
        '--obs_partial',
        type=float,
        help='Observational grid partial hyperparameter',
        default=1.0
    )

    parser.add_argument(
        '--normalize',
        type=str2bool,
        help='normalize the input or not',
        default=False
    )

    parser.add_argument(
        '--obserr',
        type=float,
        help='the observation error deviation',
        default = 1.0
    )

    parser.add_argument(
        '--da_Lag',
        type=int,
        help='Data assimilation window length',
        default=0
    )

    return parser


if __name__ == '__main__':
    parser = prepare_parser()
    args = parser.parse_args()
    res_dir = args.res_dir
    pretrain_dir = args.pretrain_dir
    system = args.system
    years = args.years
    mlda_name = args.mlda_name
    da_model = args.da_model
    pred_len = args.pred_len
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
    normalize = args.normalize
    obserr = args.obserr
    da_Lag = args.da_Lag

    eval_cycles(res_dir, pretrain_dir, system, years, mlda_name,
                da_model, pred_len, N_trials, dim, da_method,
                da_N, da_Inf, da_Bx, da_rot, da_loc_rad, da_xN,
                 obs_partial, normalize, obserr, da_Lag)