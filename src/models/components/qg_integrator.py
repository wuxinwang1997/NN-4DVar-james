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
from src.models.components.qg_kernel import qg_flux 

class integrator(nn.Module):
    def __init__(self, scheme, F, NX1, NY1, MREFIN, H1, NU1, NU2, NCYC, M, N, dx, dy, r, rkb, rkh ,rkh2, dt, CURLT, device):
        super(integrator, self).__init__()
        self.scheme = scheme
        self.dt = dt
        self.device = device
        self.qg_flux = qg_flux(F, NX1, NY1, MREFIN, H1, NU1, NU2, NCYC, M, N, dx, dy, r, rkb, rkh ,rkh2, CURLT, device)

    # def qg_step_1storder(self, t, PSI, Q):
    #     PSI, QFLUX = self.qg_flux(t, PSI, Q)
    #     Q = Q + self.dt * QFLUX
    #     t = t + self.dt
    #     return t, Q, PSI

    def qg_step_2ndorder(self, t, PSI, Q):
        PSI, QFLUX = self.qg_flux(t, PSI, Q)
        Q2 = Q + (0.5 * self.dt) * QFLUX
        Q3 = Q + self.dt * QFLUX
        PSI, QFLUX = self.qg_flux(t + 0.5 * self.dt, PSI, Q2)
        Q2 = Q2 + (0.5 * self.dt) * QFLUX

        t = t + self.dt
        Q = 2 * Q2 - Q3
        return t, Q, PSI

    def qg_step_rk4(self, t, PSI, Q):
        # Given vorticity Q, this call calculates its flux QFLUX1. 
        # Solves for PSI1 as a by-product, using PSI as the first guess
        # 
        PP, QFLUX1 = self.qg_flux(t, PSI, Q)
        tt = t + 0.5
        Q2 = Q + (0.5 * self.dt) * QFLUX1
        PSI, QFLUX2 = self.qg_flux(tt, PP, Q2)
        Q3 = Q + (0.5 * self.dt) * QFLUX2
        PP, QFLUX3 = self.qg_flux(tt, PSI, Q3)
        Q4 = Q + self.dt * QFLUX3
        tt = t + self.dt
        PSI, QFLUX4 = self.qg_flux(tt, PP, Q4)

        t = t + self.dt
        Q = Q + (QFLUX1 + 2 * (QFLUX2 + QFLUX3) + QFLUX4) * (self.dt / 6)
        return t, Q, PSI

    def qg_step_dp5(self, t, PSI, Q):
        PP, QFLUX1 = self.qg_flux(t, PSI, Q)
        tt = t + 0.2 * self.dt
        QQ = Q + (0.2 * self.dt) * QFLUX1
        PSI, QFLUX2 = self, qg_flux(tt, PP, QQ)
        tt = t + 0.3 * self.dt
        QQ = Q + ((3 / 40) * self.dt) * QFLUX1 + ((9 / 40) * self.dt) * QFLUX2
        PP, QFLUX3 = self.qg_flux(tt, PSI, QQ)
        tt = t + 0.8 * self.dt
        QQ = Q + ((44 / 45) * self.dt) * QFLUX1 - ((56 / 15) * self.dt) * QFLUX2 + ((32 / 9) * self.dt) * QFLUX3
        PSI, QFLUX4 = self.qg_flux(tt, PP, QQ)
        tt = t + (8 / 9) * self.dt
        QQ = Q + ((19372 / 6561)  * self.dt) * QFLUX1 \
            - ((25360 / 2187) * self.dt) * QFLUX2 \
            + ((64448 / 6561) * self.dt) * QFLUX3 \
            - ((212 / 729) * self.dt) * QFLUX4
        PP, QFLUX5 = self.qg_flux(tt, PSI, QQ)
        tt = t + self.dt
        QQ = Q + ((9017 / 3168) * self.dt) * QFLUX1 \
            - ((355 / 33) * self.dt) * QFLUX2 \
            + ((46732 / 5247) * self.dt) * QFLUX3 \
            + ((49 / 176) * self.dt) * QFLUX4 \
            - ((5103 / 18656) * self.dt) * QFLUX5
        PSI, QFLUX2 = self.qg_flux(tt, PP, QQ)

        t = t + self.dt
        Q = Q + ((35 / 384) * self.dt) * QFLUX1 \
            + ((500 / 1113) * self.dt) * QFLUX3 \
            + ((125 / 192) * self.dt) * QFLUX4 \
            - ((2187 / 6784) * self.dt) * QFLUX5 \
            + ((11 / 84) * self.dt) * QFLUX2

        return t, Q, PSI

    def forward(self, t, PSI, Q):
        if self.scheme == '2ndorder':
            return self.qg_step_2ndorder(t, PSI, Q)
        elif self.scheme == 'rk4':
            return self.qg_step_rk4(t, PSI, Q)
        elif self.scheme == 'dp5':
            return self.qg_step_dp5(t, PSI, Q)
        else:
            ValueError('unknown scheme ' + self.scheme) 