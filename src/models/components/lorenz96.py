import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from src.utils.model_utils import rk4_torch
import dapper
from dapper.tools.localization import nd_Id_localization
from dapper.mods.utils import name_func
from src.utils.utils import direct_obs_matrix, ens_compatible
torch.set_default_dtype(torch.float32)

class Lorenz96_torch(nn.Module):
    def __init__(self, force, device='cpu'):
        super(Lorenz96_torch, self).__init__()
        self.force = force
        self.device = device

    def shift(self, x, n):
        return torch.roll(x, shifts=-n, dims=-1)
    
    def dxdt_autonomous(self, x):
        return (self.shift(x, 1)-self.shift(x, -2))*self.shift(x, -1) - x

    def dxdt(self, x):
        return self.dxdt_autonomous(x) + self.force

    def step(self, x0, t, dt):
        return rk4_torch(lambda x, t: self.dxdt(x), x0, torch.nan, dt)

    def forward(self, x0, t, dt):
        if len(x0.shape) == 1:
            if isinstance(x0, np.ndarray):
                x0 = torch.from_numpy(np.expand_dims(np.expand_dims(x0, axis=0), axis=1)).to(self.device, dtype=torch.float32)
            else:
                x0 = np.unsqueeze(np.expand_dims(x0, axis=0), axis=1)
        if len(x0.shape) == 2:
            if isinstance(x0, np.ndarray):
                x0 = torch.from_numpy(np.expand_dims(x0, axis=1)).to(self.device, dtype=torch.float32)
            else:
                x0 = torch.unsqueeze(x0, 1)
        if len(x0.shape) == 3:
            if isinstance(x0, np.ndarray):
                x0 = torch.from_numpy(x0).to(self.device, dtype=torch.float32)
        assert len(x0.shape) == 3 # (N, J+1, K), J 'channels', K locations
        return torch.squeeze(self.step(x0, t, dt), dim=1)
    
def Lorenz96_partial_Id_Obs(Nx, obs_inds, jj):

    Ny = len(jj)
    # def linear(x, t): return direct_obs_matrix(Nx, obs_inds(t))
    localizer = nd_Id_localization([Nx,], obs_inds=obs_inds, periodic=False)

    @name_func(f"Direct obs. at {obs_inds}")
    @ens_compatible
    def model(x, t):
        return x[obs_inds(t)]

    Obs = {
        'M': Ny,
        'model': model,
        'localizer': localizer,
        # 'linear': linear
    }

    return Obs
