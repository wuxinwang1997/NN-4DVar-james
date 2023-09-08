import numpy as np
import os
import sys
sys.path.append('.')
import dapper as dpr
import dapper.da_methods as da
import dapper.tools.progressbar as pb
import dapper.mods as modelling
import dapper.mods.Lorenz96 as Lorenz96
from dapper.tools.localization import nd_Id_localization
import argparse
from src.utils.utils import str2bool
import xarray as xr
import matplotlib.pyplot as plt


def plot_search_param(data_dir, da_method):
    gts, xbs, rmse_xbs, xas, rmse_xas, obss = {}, {}, {}, {}, {}, {}
    nums = 0
    rmses, files = [], []
    for filewalks in os.walk(data_dir):
        for file in filewalks[2]:
            if da_method in file and 'gt' in file:
                gts[file[:-3]] = xr.open_mfdataset(f'{data_dir}/{file}')
            if da_method in file and 'xb' in file:
                xbs[file[:-3]] = xr.open_mfdataset(f'{data_dir}/{file}')
            if da_method in file and 'rmse_xb' in file and 'Bx1.0' not in file:
                nums += 1
                files.append(file[7:-3])
                rmse_xbs[file[:-3]] = xr.open_mfdataset(f'{data_dir}/{file}')
            if da_method in file and 'xa' in file:
                xas[file[:-3]] = xr.open_mfdataset(f'{data_dir}/{file}')
            if da_method in file and 'rmse_xa' in file  and 'Bx1.0' not in file:
                rmse_xas[file[:-3]] = xr.open_mfdataset(f'{data_dir}/{file}')
            if da_method in file and 'obs' in file:
                obss[file[:-3]] = xr.open_mfdataset(f'{data_dir}/{file}')
    
    rmses.append(rmse_xbs)
    rmses.append(rmse_xas)
    fig = plt.figure(figsize=(20, 10))
    for i in range(2):
        ax = fig.add_subplot(2, 1, i+1)
        for j in range(nums):
            if i == 0:
                ax.plot(rmses[i][f'rmse_xb{files[j]}']['lead_time'], rmses[i][f'rmse_xb{files[j]}']['x'], label=f'{files[j][1:]}')
            else:
                ax.plot(rmses[i][f'rmse_xa{files[j]}']['lead_time'], rmses[i][f'rmse_xa{files[j]}']['x'], label=f'{files[j][1:]}')

        ax.legend(loc='upper left')

    plt.tight_layout()
    plt.savefig(f'{data_dir}/{da_method}_search_param.jpg',dpi=300)

    


def prepare_parser():
    parser = argparse.ArgumentParser(description='Generate Background')
    parser.add_argument(
        '--data_dir',
        type=str,
        help='path to save the dataset',
        default='./data/search_param'
    )

    parser.add_argument(
        '--da_method',
        type=str,
        help = 'data assimilation method utilized to generate dataset',
        default='DEnKF'
    )

    return parser


if __name__ == '__main__':
    parser = prepare_parser()
    args = parser.parse_args()
    data_dir = args.data_dir
    da_method = args.da_method


    plot_search_param(data_dir, da_method)