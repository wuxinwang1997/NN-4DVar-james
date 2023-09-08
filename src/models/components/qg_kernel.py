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

import torch 
import torch.nn as nn
import numpy as np

class arakawa(nn.Module):
    def __init__(self, dx, dy, device):
        super(arakawa, self).__init__()
        self.dx = dx
        self.dy = dy
        self.device = device

    def forward(self, A, B):
        n1 = A.shape[-2] - 1
        m1 = A.shape[-1] - 1
        JACOBIAN = torch.zeros(size=A.shape).to(self.device)
        JACOBIAN[:, :, 1:n1, 1:m1] = (A[:, :, 1:n1, 0:m1-1] - A[:, :, 0:n1-1, 1:m1]) * B[:, :, 0:n1-1, 0:m1-1] +\
                (A[:, :, 0:n1-1, 0:m1-1] + A[:, :, 1:n1, 0:m1-1] - A[:, :, 0:n1-1, 2:m1+1] - A[:, :, 1:n1, 2:m1+1]) * B[:, :, 0:n1-1, 1:m1] +\
                (A[:, :, 0:n1-1, 1:m1] - A[:, :, 1:n1, 2:m1+1]) * B[:, :, 0:n1-1, 2:m1+1] +\
                (A[:, :, 2:n1+1, 0:m1-1] + A[:, :, 2:n1+1, 1:m1] - A[:, :, 0:n1-1, 0:m1-1] - A[:, :, 0:n1-1, 1:m1]) * B[:, :, 1:n1, 0:m1-1] +\
                (A[:, :, 0:n1-1, 1:m1] + A[:, :, 0:n1-1, 2:m1+1]- A[:, :, 2:n1+1, 1:m1] - A[:, :, 2:n1+1, 2:m1+1]) * B[:, :, 1:n1, 2:m1+1] +\
                (A[:, :, 2:n1+1, 1:m1] - A[:, :, 1:n1, 0:m1-1]) * B[:, :, 2:n1+1, 0:m1-1] +\
                (A[:, :, 1:n1, 2:m1+1] + A[:, :, 2:n1+1, 2:m1+1] - A[:, :, 1:n1, 0:m1-1] - A[:, :, 2:n1+1, 0:m1-1]) * B[:, :, 2:n1+1, 1:m1] +\
                (A[:, :, 1:n1, 2:m1+1] - A[:, :, 2:n1+1, 1:m1]) * B[:, :, 2:n1+1, 2:m1+1]

        JACOBIAN[:, :, 1:n1, 1:m1] = JACOBIAN[:, :, 1:n1, 1:m1] / (12 * self.dx * self.dy)
        return JACOBIAN

class laplacian(nn.Module):
    def __init__(self, dx, dy, device):
        super(laplacian, self).__init__()
        self.dx2 = dx * dx
        self.dy2 = dy * dy
        self.device = device
    
    def forward(self, A):
        n = A.shape[-2]
        M = A.shape[-1]

        L = torch.zeros(size=A.shape).to(self.device)
        
        L[:, :, 1:n-1, 1:M-1] = \
            (A[:, :, 1:n-1, 0:M-2] + A[:, :, 1:n-1, 2:M]) / self.dx2 \
            + (A[:, :, 0:n-2, 1:M-1] + A[:, :, 2:n, 1:M-1]) / self.dy2 \
            - A[:, :, 1:n-1, 1:M-1] * (2 / self.dx2 + 2 / self.dy2)

        # L = 0 on the boundary
        L[:, :, :, 0] = 0
        L[:, :, :, M-1] = 0
        L[:, :, 0, :] = 0
        L[:, :, n-1, :] = 0

        return L

class Relax_Conv(nn.Module):
    def __init__(self, F, h, device):
        super(Relax_Conv, self).__init__()
        self.conv = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=3, bias=False)
        self.conv.weight.data = torch.from_numpy(np.float32(1 / (4+F*h**2)) * np.asarray([[[[0, 1, 0],
                                                                                            [1, 0, 1],
                                                                                            [0, 1, 0]],
                                                                                           [[0, 0, 0],
                                                                                            [0, -1, 0],
                                                                                            [0, 0, 0]],]])).to(device, dtype = torch.float32)
        self.conv.to(device)
    def forward(self, x):
        x = self.conv(x)
        x = nn.functional.pad(input=x, pad=(1, 1, 1, 1), mode='constant', value=0)
        return x

class Rescal_Conv(nn.Module):
    def __init__(self, F, h, device):
        super(Rescal_Conv, self).__init__()
        self.conv = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=5, stride=2, bias=False)
        self.conv.weight.data = torch.from_numpy(4 * np.asarray([[[[0, 0, 0, 0, 0],
                                                                   [0, 0, -1, 0, 0],
                                                                   [0, -1, 4+F*h**2, -1, 0],
                                                                   [0, 0, -1, 0, 0],
                                                                   [0, 0, 0, 0, 0]],
                                                                  [[0, 0, 0, 0, 0],
                                                                   [0, 0, 0, 0, 0],
                                                                   [0, 0, 1, 0, 0],
                                                                   [0, 0, 0, 0, 0],
                                                                   [0, 0, 0, 0, 0]]]])).to(device, dtype = torch.float32)
        self.conv.to(device)
    def forward(self, x):
        x = self.conv(x)
        x = nn.functional.pad(input=x, pad=(1, 1, 1, 1), mode='constant', value=0)
        return x

class helmholtz(nn.Module):
    def __init__(self, F, NX1, NY1, MREFIN, H1, NU1, NU2, NCYC, MX, NY, device):
        super(helmholtz, self).__init__()
        #
        # Solves helmholtz equation:
        # (\nabla^2 - zk2) P  = F, with Dirichlet B.C. The B.C. and the initial
        # guess for the solution are given in the array G.
        #
        # The dimension of the arrays should be:
        # (NX, MY) = (p * 2^(M - 1) + 1, q * 2^(M - 1) + 1).
        # p and q are small integers, equal to the number of grid boxes in the
        # coarsest grid used in the multi-grid solution.
        #
        # Note that number of boxes on a side of the grid is equal to the number
        # of grid points on this side - 1.
        #
        # In the code below p = NX1; q = NX2.                                       **
        # M is the finest level of the multigrid. M=1 means use (p,q)     **
        # boxes; M=2 means use twice as many boxes on each lide of grid.  **
        #
        # H1 is the delx for the coarsest grid. (NU1,NU2) should be (1,2). **
        # NCYCL should be 3-5 for a bad initial guess, and 1 for a good   **
        # initial guess.                                                  **
        # The dimension of the work array q should be NX * MY * 3.          **
        #
        self.F = F
        self.NX1 = NX1
        self.NY1 = NY1
        self.MREFIN = MREFIN
        self.H1 = H1
        self.NU1 = NU1
        self.NU2 = NU2
        self.NCYC = NCYC
        self.MX = MX
        self.NY = NY
        self.device = device
        self.h = [self.H1 / 2**(MREFIN-k-1) for k in range(MREFIN)]  
        self.relax_convs = [Relax_Conv(self.F, self.h[k], self.device) for k in range(MREFIN)]
        self.rescal_convs = [Rescal_Conv(self.F, self.h[k], self.device) for k in range(MREFIN)]


    def intadd(self, psis, kc, kf):

        tmp = nn.functional.interpolate(input=psis[kc], size=psis[kf].shape[-2:], mode='bilinear', align_corners=True)
        psis[kf] += torch.squeeze(tmp, dim=1)
        return psis

    def forward(self, PSIGUESS, Q):
        bs, ch, w, h = PSIGUESS.shape
        psis = [torch.zeros(bs, ch, int(np.ceil(w/2**k)), int(np.ceil(h/2**k)), dtype=torch.float32).to(self.device) for k in range(self.MREFIN)]
        qs = [torch.zeros(bs, ch, int(np.ceil(w/2**k)), int(np.ceil(h/2**k)), dtype=torch.float32).to(self.device) for k in range(self.MREFIN)]
        psis[0] = PSIGUESS * self.h[0]**0
        qs[0] = Q * self.h[0]**2
        for ic in range(0, self.NCYC):
            for k in range(0, self.MREFIN):
                if k != 0:
                    psis[k][:, :, :, :] = 0
                for ir in range(0, self.NU1):
                    psis[k] = self.relax_convs[k](torch.concat((psis[k], qs[k]), dim=1)) #(k, k + self.MREFIN, self.F, q1, q2)
                if (k < self.MREFIN-1):
                    qs[k+1] = self.rescal_convs[k](torch.concat((psis[k], qs[k]), dim=1)) #(k, k + self.MREFIN, k + self.MREFIN - 1, self.F, q1, q2)
            for k in range(0, self.MREFIN):
                for ir in range(0, self.NU2):
                    psis[self.MREFIN-k-1] = self.relax_convs[self.MREFIN-k-1](torch.concat((psis[self.MREFIN-k-1], qs[self.MREFIN-k-1]), dim=1)) #(k, k + self.MREFIN, self.F, q1, q2)
                if (k > 0):
                    psis = self.intadd(psis, k, k-1)
        PSI = psis[0] 
        return PSI

class calc_psi(nn.Module):
    def __init__(self, F, NX1, NY1, MREFIN, H1, NU1, NU2, NCYC, MX, NY, device):
        super(calc_psi, self).__init__()
        self.device = device
        self.helmholtz = helmholtz(F, NX1, NY1, MREFIN, H1, NU1, NU2, NCYC, MX, NY, self.device)

    def forward(self, PSIGUESS, Q):
        PSI = self.helmholtz(PSIGUESS, Q)
        return PSI

class qg_flux(nn.Module):
    def __init__(self, F, NX1, NY1, MREFIN, H1, NU1, NU2, NCYC, M, N, dx, dy, r, rkb, rkh, rkh2, CURLT, device):
        super(qg_flux, self).__init__()
        self.device = device
        self.calc_psi = calc_psi(F, NX1, NY1, MREFIN, H1, NU1, NU2, NCYC, M, N, self.device)
        self.jacobian = arakawa(dx, dy, self.device)
        self.laplacian = laplacian(dx, dy, self.device)
        self.M = M
        self.N = N
        self.dx = dx
        self.dy = dy
        self.r = r
        self.rkb = rkb
        self.rkh = rkh
        self.rkh2 = rkh2
        self.CURLT = torch.unsqueeze(torch.from_numpy(CURLT), dim=0).to(self.device) 

    def forward(self, t, PSIGUESS, Q):
        PSI = self.calc_psi(PSIGUESS, Q)
        JACOBIAN = self.jacobian(PSI, Q)
        ZETA = self.laplacian(PSI)
        ZETA2 = self.laplacian(ZETA)
        ZETA4 = self.laplacian(ZETA2)

        QFLUX = -self.r*JACOBIAN - self.rkb*ZETA + self.rkh*ZETA2 - self.rkh2*ZETA4 + self.CURLT
        QFLUX[:, :, 1:self.M-1, 1: self.N-1] = QFLUX[:, :, 1:self.M-1, 1:self.N-1] -(0.5/self.dx)*(PSI[:, :, 1:self.M-1, 2:self.N]-PSI[:, :, 1:self.M-1, 0:self.N-2])
        QFLUX[:, :, :, 0] = 0
        QFLUX[:, :, :, self.N-1] = 0
        QFLUX[:, :, 0, :] = 0
        QFLUX[:, :, self.M-1, :] = 0

        return PSI, QFLUX