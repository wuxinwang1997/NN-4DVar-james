# Copyright (C) 2022  Wuxin Wang

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

import sys
sys.path.append('.')
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from src.models.components.qg_integrator import integrator
from src.models.components.qg_kernel import laplacian, calc_psi
import matplotlib as mpl
# from .QG import interface
import dapper.mods as modelling
import dapper.tools.liveplotting as LP
torch.set_default_dtype(torch.float32)
from src.utils.qg_tools import parameters_read, default_prms
from src.utils.utils import with_recursion

class torch_qg(nn.Module):
    def __init__(self, 
                 dt: float = 0.125,
                 dtout: float = 0.25, # 0.5
                 RKB: float = 0,
                 RKH: float = 0,
                 RKH2: float = 2e-12,
                 F: float = 1600,
                 R: float = 1e-05,
                 scheme: str = '2ndorder',
                 tend: float = 0,
                 verbose: int = 0,
                 restart: int = 0,
                 MREFIN: int = 7, # 7, 
                 NX1: int = 2, 
                 NY1: int = 2,
                 device: str = 'cpu'):
        super(torch_qg, self).__init__()
        self.device = device
        self.dt = dt
        self.dtout = dtout
        self.RKB = RKB
        self.RKH = RKH
        self.RKH2 = RKH2
        self.F = F
        self.R = R
        self.scheme = scheme
        self.tend = tend
        self.verbose = verbose
        self.restart = restart
        self.MREFIN = MREFIN
        self.NX1 = NX1
        self.NY1 = NY1
        # grids of the field in X and Y
        self.M = self.NX1 * 2 ** (self.MREFIN - 1) + 1
        self.N = self.NY1 * 2 ** (self.MREFIN - 1) + 1
        self.shape = (self.M, self.N)
        # Roberts filter coefficient for the leap-frog scheme 
        self.rf_coeff = 0.1
        # dx and dy of the field
        self.dx = 1 / (self.M - 1)
        self.dy = 1 / (self.N - 1)

        self.CURLT = np.zeros(shape=self.shape).astype(np.float32)
        for i in range(0, self.M):
            self.CURLT[i, :] = -2*np.pi*np.sin(2*np.pi*i/(self.M-1))
        self.laplacian = laplacian(self.dx, self.dy, self.device)
        self.calc_psi = calc_psi(self.F, self.NX1, self.NY1, self.MREFIN, 1 / self.NX1, 1, 2, 3, self.M, self.N, self.device)
        self.integrator = integrator(self.scheme, self.F, self.NX1, self.NY1, 
                                    self.MREFIN, 1 / self.NX1, 1, 2, 3, self.M, self.N, self.dx, 
                                    self.dy, self.R, self.RKB, self.RKH, 
                                    self.RKH2, self.dt, self.CURLT, self.device)
    
    def vec2tensor(self, x): 
        return x.reshape(-1, 1, self.M, self.N)
    
    def tensor2vec(self, X): 
        return torch.flatten(X, start_dim=1)

    def step(self, x0, t, dt):
        """Step a single state vector."""
        # Coz fortran.step() reads dt (dtout) from prms file:
        assert self.dtout == dt
        # Coz Fortran is typed.
        assert isinstance(t, float)
        # QG is autonomous, but Fortran doesn't like nan/inf.
        assert np.isfinite(t)
        # Copy coz Fortran will modify in-place.
        if isinstance(x0, np.ndarray):
            psi = torch.from_numpy(self.vec2tensor(x0)).to(self.device, dtype=torch.float32)
        elif len(x0.shape) != 4:
            psi = self.vec2tensor(x0)
        else:
            psi = x0
        tstop = t + self.dtout 
        q = self.laplacian(psi)
        q = q - self.F * psi
        while (t < tstop):
            t, q, psi = self.integrator(t, psi, q)
        psi = self.calc_psi(psi, q)
        x = self.tensor2vec(psi)
        return x

    def forward(self, E, t, dt):
        return self.step(E, t, dt)

#########################
# Free run
#########################
def gen_sample(model, shape, nSamples, SpinUp, Spacing):
    simulator = with_recursion(model.step, prog="Simulating")
    K         = SpinUp + nSamples*Spacing
    Nx        = np.prod(shape)  # total state length
    sample    = simulator(np.zeros(Nx), K, 0.0, model.dtout)
    return sample[SpinUp::Spacing]

qg = torch_qg()

sample_filename = modelling.rc.dirs.samples/f'QG_MREFIN{qg.MREFIN}_scheme{qg.scheme}_samples.npz'
if (not sample_filename.is_file()) and ("pdoc" not in sys.modules):
    print('Did not find sample file', sample_filename,
          'for experiment initialization. Generating...')
    sample = gen_sample(qg, qg.shape, 400, 700, 10)
    np.savez(sample_filename, sample=sample)

#########################
# Liveplotting
#########################
cm = mpl.colors.ListedColormap(0.85*mpl.cm.jet(np.arange(256)))
center = qg.M*int(qg.M/2) + int(0.5*qg.N)

def square(x): return x.reshape(qg.shape[::-1])
def ind2sub(ind): return np.unravel_index(ind, qg.shape[::-1])

def LP_setup(jj=None): return [
    (1, LP.spatial2d(square, ind2sub, jj, cm)),
    (0, LP.spectral_errors),
    (0, LP.sliding_marginals(dims=center+np.arange(4))),
]