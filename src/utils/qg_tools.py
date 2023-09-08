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


# This is the place to change the resolution of the model, 
# by changing the parameter MREFIN:
# To make the ocean a rectangular (at youre own risk!) change nx1,
# ny1 to (2, 4), and adjust domain size (Lx. Ly) 

import numpy as np
import pandas as pd
import matplotlib as mpl
import dapper
import dapper.tools.liveplotting as LP

#########################
# Model
#########################
default_prms = dict(
    # These parameters may be interesting to change.
    dtout        = 5.0,      # dt for output to DeepDA.
    dt           = 1.25,     # dt used internally by Fortran. CFL = 2.0
    RKB          = 0,        # bottom     friction
    RKH          = 0,        # horizontal friction
    RKH2         = 2.0e-12,  # horizontal friction, biharmonic
    F            = 1600,     # Froud number
    R            = 1.0e-5,   # ≈ Rossby number
    scheme       = "'rk4'",  # One of (2ndorder, rk4, dp5)
    # Do not change the following:
    tend         = 0,        # Only used by standalone QG
    verbose      = 0,        # Turn off
    rstart       = 0,        # Restart: switch
    restartfname = "''",     # Restart: read file
    outfname     = "''",     # Restart: write file
)

def parameters_read(prmfname, MREFIN=7, NX1=2, NY1=2):
    parameters = dict()
    df = pd.read_csv(prmfname, sep='[\t]', header=1, engine='python')
    parameters['dtout'] = float(df.loc[0, '&parameters'].split('=')[-1])
    parameters['dt'] = float(df.loc[1, '&parameters'].split('=')[-1])
    parameters['RKB'] = float(df.loc[2, '&parameters'].split('=')[-1])
    parameters['RKH'] = float(df.loc[3, '&parameters'].split('=')[-1])
    parameters['RKH2'] = float(df.loc[4, '&parameters'].split('=')[-1])
    parameters['F'] = float(df.loc[5, '&parameters'].split('=')[-1])
    parameters['R'] = float(df.loc[6, '&parameters'].split('=')[-1])
    parameters['scheme'] = df.loc[7, '&parameters'].split('=')[-1].split("\'")[1]
    parameters['tend'] = float(df.loc[8, '&parameters'].split('=')[-1])
    parameters['verbose'] = int(df.loc[9, '&parameters'].split('=')[-1])
    parameters['restart'] = float(df.loc[10, '&parameters'].split('=')[-1])
    parameters['restartfname'] = df.loc[11, '&parameters'].split('=')[-1].split("\'")[1]
    parameters['outfname'] = df.loc[12, '&parameters'].split('=')[-1].split("\'")[1]
    parameters['MREFIN'] = MREFIN
    parameters['NX1'] = NX1
    parameters['NY1'] = NY1
    parameters['M'] = parameters['NX1'] * 2 ** (parameters['MREFIN'] - 1) + 1
    parameters['N'] = parameters['NY1'] * 2 ** (parameters['MREFIN'] - 1) + 1
    parameters['PI'] = np.pi
    parameters['rf_coeff'] = 0.1 #Roberts filter coefficient for the leap-frog scheme 
    parameters['dx'] = 1 / (parameters['M'] - 1)
    parameters['dy'] = 1 / (parameters['N'] - 1)
    if parameters['verbose'] > 0:
        print(parameters)

    CURLT = np.zeros(shape=(parameters['M'], parameters['N']), dtype=np.float32)
    for i in range(0, parameters['M']):
        CURLT[i, :] = -2*parameters['PI']*np.sin(2*parameters['PI']*i/(parameters['M']-1))

    parameters['CURLT'] = CURLT

    return parameters

# However, FOR PRINTING/PLOTTING PURPOSES, the y-axis should be vertical
# [imshow(mat) uses the same orientation as print(mat)].
def square(self, x, shape): 
    return x.reshape(shape[::-1])

def ind2sub(self, ind, shape): 
    return np.unravel_index(ind, shape[::-1])


def LP_setup(nx, ny, jj=None): 
    #########################
    # Liveplotting
    #########################
    cm = mpl.colors.ListedColormap(0.85*mpl.cm.jet(np.arange(256)))
    center = nx*int(nx/2) + int(0.5*ny)
    return [
        (1, LP.spatial2d(square, ind2sub, jj, cm)),
        (0, LP.spectral_errors),
        (0, LP.sliding_marginals(dims=center+np.arange(4))),
    ]
