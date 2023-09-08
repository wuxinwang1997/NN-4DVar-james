import torch
from torch.utils.data import Dataset
from torchvision.transforms import transforms
import xarray as xr
from pathlib import Path
import pickle
import numpy as np
import copy
from src.utils.tools import gaussian_perturb_np

class L96_Dataset(Dataset):
    def __init__(self, 
                 data_dir, 
                 da_method,
                 years,
                 N_trials,
                 dim,
                 da_N,
                 da_Inf,
                 da_Bx,
                 da_rot,
                 da_loc_rad,
                 obs_partial,
                 da_Lag,
                 obs_num,
                 pred_len,
                 normalize,
                 train):

        gt = xr.open_mfdataset(
            f'{data_dir}/train/L96_gt_{da_method}_T{years}_N_trials{N_trials}_N{dim}_daN{da_N}_daInf{da_Inf}_daBx{da_Bx}_darot{da_rot}_dalocrad{da_loc_rad}_obspartial{obs_partial}_daLag{da_Lag}.nc', combine='by_coords')
        xb = xr.open_mfdataset(
            f'{data_dir}/train/L96_xb_{da_method}_T{years}_N_trials{N_trials}_N{dim}_daN{da_N}_daInf{da_Inf}_daBx{da_Bx}_darot{da_rot}_dalocrad{da_loc_rad}_obspartial{obs_partial}_daLag{da_Lag}.nc', combine='by_coords')
        xa = xr.open_mfdataset(
            f'{data_dir}/train/L96_xa_{da_method}_T{years}_N_trials{N_trials}_N{dim}_daN{da_N}_daInf{da_Inf}_daBx{da_Bx}_darot{da_rot}_dalocrad{da_loc_rad}_obspartial{obs_partial}_daLag{da_Lag}.nc', combine='by_coords')
        obs = xr.open_mfdataset(
            f'{data_dir}/train/L96_obs_{da_method}_T{years}_N_trials{N_trials}_N{dim}_daN{da_N}_daInf{da_Inf}_daBx{da_Bx}_darot{da_rot}_dalocrad{da_loc_rad}_obspartial{obs_partial}_daLag{da_Lag}.nc', combine='by_coords')
        obs_all = xr.open_mfdataset(
            f'{data_dir}/train/L96_obs_all_{da_method}_T{years}_N_trials{N_trials}_N{dim}_daN{da_N}_daInf{da_Inf}_daBx{da_Bx}_darot{da_rot}_dalocrad{da_loc_rad}_obspartial{obs_partial}_daLag{da_Lag}.nc',
            combine='by_coords')
        stats = np.load(f'{data_dir}/train/L96_stats_{da_method}_T{years}_N_trials{N_trials}_N{dim}_daN{da_N}_daInf{da_Inf}_daBx{da_Bx}_darot{da_rot}_dalocrad{da_loc_rad}_obspartial{obs_partial}_daLag{da_Lag}.npz')

        self.obs_num = obs_num
        self.dim = dim
        self.pred_len = pred_len
        self.normalize = normalize
        self.train = train

        if self.train:
            if self.pred_len > 0:
                self.obs = obs.sel(trial=0)['x'].values[:-self.pred_len, :].astype(np.float32)
            else:
                self.obs = obs.sel(trial=0)['x'].values.astype(np.float32)
            self.xb = xb.sel(trial=0)['x'].values.astype(np.float32)
            self.gt = gt.sel(trial=0)['x'].values.astype(np.float32)
            self.xa = xa.sel(trial=0)['x'].values.astype(np.float32)
            self.obs_all = obs_all.sel(trial=0)['x'].values.reshape(self.xa.shape[0], -1, int(obs_partial*dim)).astype(np.float32)
            self.obs_idx = obs.sel(trial=0)['grid'].values.astype(np.int32)
        else:
            if self.pred_len > 0:
                self.obs = obs.sel(trial=1)['x'].values[:-self.pred_len, :].astype(np.float32)
            else:
                self.obs = obs.sel(trial=1)['x'].values.astype(np.float32)
            self.xb = xb.sel(trial=1)['x'].values.astype(np.float32)
            self.gt = gt.sel(trial=1)['x'].values.astype(np.float32)
            self.xa = xa.sel(trial=1)['x'].values.astype(np.float32)
            self.obs_all = obs_all.sel(trial=1)['x'].values.reshape(self.xa.shape[0], -1, int(obs_partial*dim)).astype(np.float32)
            self.obs_idx = obs.sel(trial=1)['grid'].values.astype(np.int32)

        self.mean_xb = stats['mean_xb']
        self.std_xb = stats['std_xb']
        self.mean_xa = stats['mean_xa']
        self.std_xa = stats['std_xa']
        self.mean_obs = stats['mean_obs']
        self.std_obs = stats['std_obs']
        #
        # self.transforms_xb = transforms.Compose(
        #     [transforms.ToTensor(), transforms.Normalize(self.mean_xb, self.std_xb)]
        # )
        #
        # self.transforms_xa = transforms.Compose(
        #     [transforms.ToTensor(), transforms.Normalize(self.mean_xa, self.std_xa)]
        # )
        #
        # self.transforms_obs = transforms.Compose(
        #     [transforms.ToTensor(), transforms.Normalize(self.mean_obs, self.std_obs)]
        # )
        #
        # self.transforms_tensor = transforms.ToTensor()

    def __len__(self):
        return self.xb.shape[0]-self.pred_len

    def __getitem__(self, idx):
        if self.normalize:
            gt = self.gt[idx:idx+1,:]
            xb = (self.xb[idx:idx+1,:] - self.mean_xb) / self.std_xb
            xa = (self.xa[idx:idx+self.pred_len+1,:] - self.mean_xa) / self.std_xa
            obs_all = np.ones((xa.shape[0], self.obs_num, self.dim)).astype(np.float32) * self.mean_obs
            obs_idx = self.obs_idx
            obs_all[:, :, obs_idx] = self.obs_all[idx, :, :]
            obs_all = (obs_all[idx, :] - self.mean_obs) / self.std_obs
            return gt, xb, xa, obs_all[0:self.obs_num,:], obs_idx, self.mean_xa*np.ones_like(gt), self.std_xa*np.ones_like(gt)
        else:
            gt = self.gt[idx:idx + self.pred_len + 1, :]
            xb = self.xb[idx:idx + self.pred_len + 1, :]
            xa = self.xa[idx:idx + self.pred_len + 1, :]
            obs_all = np.zeros((self.obs_num, self.dim)).astype(np.float32)
            obs_idx = self.obs_idx
            obs_all[:, obs_idx] = self.obs_all[idx, :, :]
            return gt, xb, xa, obs_all[0:self.obs_num,:], obs_idx, np.zeros_like(xb[:1,:]), np.ones_like(xb[:1,:])