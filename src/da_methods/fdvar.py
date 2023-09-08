"""Variational DA methods (iEnKS, 4D-Var, etc)."""

from typing import Optional

import numpy as np
import scipy.linalg as sla

import dapper
from dapper.da_methods.ensemble import hyperprior_coeffs, post_process, zeta_a
from dapper.stats import center, inflate_ens, mean0
from dapper.tools.linalg import pad0, svd0, tinv
from dapper.tools.matrices import CovMat
from dapper.tools.progressbar import progbar
import copy
import os
import sys
sys.path.append('.')
from dapper.da_methods import da_method
import torch
from src.models.components.qg import qg
torch.set_default_dtype(torch.float32)
torch.set_default_tensor_type(torch.FloatTensor)

@da_method
class var_method:
    """Declare default variational arguments."""

    Lag: int    = 1
    nIter: int  = 10
    wtol: float = 0


@var_method
class My4DVar:
    """4D-Var.

    Cycling scheme is same as in iEnKS (i.e. the shift is always 1*ko).

    This implementation does NOT do gradient decent (nor quasi-Newton)
    in an inner loop, with simplified models.
    Instead, each (outer) iteration is computed
    non-iteratively as a Gauss-Newton step.
    Thus, since the full (approximate) Hessian is formed,
    there is no benefit to the adjoint trick (back-propagation).
    => This implementation is not suited for big systems.

    Incremental formulation is used, so the formulae look like the ones in iEnKS.
    """

    B: Optional[np.ndarray] = None
    xB: float               = 1.0
    lr: float               = 1e-1
    max_iter: int           = 20
    tolerance_grad: int     = 1e-7
    tolerance_change: float = 1e-4
    history_size: int       = 20
    device: str             = 'cpu'

    def assimilate(self, HMM, xx, yy, yy_all):
        # print('start 4dvar')
        R, Ko = HMM.Obs.noise.C, HMM.tseq.Ko
        R_inv = R.inv
        R_inv = torch.from_numpy(R_inv).to(self.device, dtype=torch.float32)
        Nx = HMM.Dyn.M

        # Set background covariance. Note that it is static (compare to iEnKS).
        if isinstance(self.B, np.ndarray):
            # compare ndarray 1st to avoid == error for ndarray
            B = self.B.astype(float)
        elif self.B in (None, 'clim'):
            # Use climatological cov, estimated from truth
            if HMM.Dyn.name == 'QG':
                if not os.path.exists('/public/home/wangwuxing01/research/idea1/revision1/data/matrix'):
                    os.mkdir('/public/home/wangwuxing01/research/idea1/revision1/data/matrix')
                if not os.path.exists(f'/public/home/wangwuxing01/research/idea1/revision1/data/matrix/QG_MREFIN{qg.MREFIN}_scheme{qg.scheme}_B.npy'):
                    x_wobc = xx.reshape((-1, qg.M, qg.N))[:,1:-1,1:-1]
                    x_wobc = x_wobc.reshape((-1, (qg.M-2)*(qg.N-2)))
                    B = np.cov(x_wobc.T)
                    B = np.diag(np.diag(B))
                    np.save(f'/public/home/wangwuxing01/research/idea1/revision1/data/matrix/QG_MREFIN{qg.MREFIN}_scheme{qg.scheme}_B.npy', B)
                else:
                    B = np.load(f'/public/home/wangwuxing01/research/idea1/revision1/data/matrix/QG_MREFIN{qg.MREFIN}_scheme{qg.scheme}_B.npy')
            else:
                B = np.cov(xx.T)
        elif self.B == 'eye':
            if HMM.Dyn.name == 'QG':
                B = np.eye((qg.M-2)*(qg.N-2))
            else:
                B = np.eye(Nx)
        else:
            raise ValueError("Bad input B.")
        if self.B == 'eye':
            B_inv = torch.from_numpy(B / self.xB).to(self.device, dtype=torch.float32)
        else:
            if HMM.Dyn.name == 'QG':
                if not os.path.exists('/public/home/wangwuxing01/research/idea1/revision1/data/matrix'):
                    os.mkdir('/public/home/wangwuxing01/research/idea1/revision1/data/matrix')
                if not os.path.exists(f'/public/home/wangwuxing01/research/idea1/revision1/data/matrix/QG_MREFIN{qg.MREFIN}_scheme{qg.scheme}_B_inv.npy'):
                    B_inv = np.linalg.inv(B)
                    np.save(f'/public/home/wangwuxing01/research/idea1/revision1/data/matrix/QG_MREFIN{qg.MREFIN}_scheme{qg.scheme}_B_inv.npy', B_inv)
                else:
                    B_inv = np.load(f'/public/home/wangwuxing01/research/idea1/revision1/data/matrix/QG_MREFIN{qg.MREFIN}_scheme{qg.scheme}_B_inv.npy')
                B_inv =  torch.from_numpy(B_inv / self.xB).to(self.device, dtype=torch.float32)
            else:
                B_inv =  torch.from_numpy(np.linalg.inv(B * self.xB)).to(self.device, dtype=torch.float32)

        yy_all = torch.from_numpy(yy_all).to(self.device, dtype=torch.float32)
        yy_all.requires_grad = False

        # Init
        if isinstance(HMM.X0, dapper.tools.randvars.RV):
            x = HMM.X0.sample(1)
        else:
            x = HMM.X0.mu

        self.stats.assess(0, mu=np.squeeze(x), Cov=np.eye(Nx))

        # Loop over DA windows (DAW).
        for ko in progbar(np.arange(-1, Ko+self.Lag+1)):
            # print('start DA')
            kLag = ko-self.Lag
            DAW = range(max(0, kLag+1), min(ko, Ko) + 1)

            # forecast the firstguess
            if -1 <= kLag < Ko:
                # if kko < dko, it will forward into the starting point of the first DAW to save the firstguess
                x = torch.from_numpy(x).to(self.device, dtype=torch.float32)
                with torch.no_grad():
                    for k, t, dt in HMM.tseq.cycle(kLag+1):
                        self.stats.assess(k-1, None, 'u', mu=np.squeeze(x.detach().cpu().numpy()), Cov=np.eye(Nx))
                        x = HMM.Dyn(x, t-dt, dt)
                self.stats.assess(k, ko, 'f', mu=np.squeeze(x.detach().cpu().numpy()), Cov=np.eye(Nx))

            # Assimilation (if ∃ "not-fully-assimlated" Obs).
            if 0 <= ko <= Ko:
                # Init iterations.
                x0 = x.detach() # make xa can be optimized by the optimizer
                xa = copy.deepcopy(x0)
                xa.requires_grad = True
                # LBFGS optimizer
                opt = torch.optim.LBFGS(params=[xa],
                            lr=self.lr,
                            max_iter=self.max_iter,
                            max_eval=-1,
                            tolerance_grad=self.tolerance_grad,
                            tolerance_change=self.tolerance_change,
                            history_size=self.history_size,
                            line_search_fn='strong_wolfe')
                
                # cost function of 4DVar
                def closure():
                    x_tmp = xa
                    if HMM.Dyn.name == 'QG':
                        x_tmp_wobc = x_tmp.reshape((-1, qg.M, qg.N))[:,1:-1,1:-1]
                        x0_wobc = x0.reshape((-1, qg.M, qg.N))[:,1:-1,1:-1]
                        x_tmp_wobc = x_tmp_wobc.reshape((-1, (qg.M-2)*(qg.N-2)))
                        x0_wobc = x0_wobc.reshape((-1, (qg.M-2)*(qg.N-2)))
                        loss_xb = 0.5 * (x_tmp_wobc - x0_wobc) @ B_inv @ (x_tmp_wobc - x0_wobc).permute(1, 0)
                    else:
                        loss_xb = 0.5 * (x_tmp - x0) @ B_inv @ (x_tmp - x0).permute(1, 0)
                    dy = HMM.Obs(x_tmp, HMM.tseq.tt[ko]) - yy_all[ko*HMM.tseq.dko,:]
                    loss_y = 0.5 * dy @ R_inv @ dy.permute(1, 0)
                    for kCycle in DAW:
                        for k, t, dt in HMM.tseq.cycle(kCycle):
                            if k % HMM.tseq.dko != 0:
                                x_tmp = HMM.Dyn(x_tmp, t-dt, dt)
                                dy = HMM.Obs(x_tmp, t) - yy_all[ko*HMM.tseq.dko+k%HMM.tseq.dko,:]
                                loss_y += 0.5 * dy @ R_inv @ dy.permute(1, 0)
                    
                    loss = loss_xb + loss_y
                    loss.backward()
                    return loss

                for iteration in np.arange(self.nIter):
                    opt.step(closure)
                    opt.zero_grad()

                # Assess (analysis) self.stats.
                x = xa
                self.stats.assess((ko+1)*HMM.tseq.dko, ko, 'a', mu=np.squeeze(x.detach().cpu().numpy()), Cov=np.eye(Nx))
                self.stats.iters[ko] = iteration + 1

            if isinstance(x, torch.Tensor):
                x = x.detach().cpu().numpy()

        self.stats.assess(k, Ko, 'us', mu=np.squeeze(x), Cov=np.eye(Nx))
