"""This uses model simulations with observations to produce validation metrics""" 
import sys
import xarray as xr
import pandas as pd
import numpy as np
import netCDF4 as nc
import time
import dateutil

sys.path.append('../../')

from constants import make_icestupa
from config import time_start, time_end, icestupa_name, obs_type
from utilities.aws2cosipy.aws2cosipyConfig import *
from cosipy.modules.radCor import *

if __name__ == "__main__":
    icestupa = icestupa_name
    ds = xr.open_dataset("../../data/output/" + icestupa + ".nc")
    ds = ds.drop_vars(['lat','lon'])
    ds['TS'] -= 273.16
    ds['TBULK'] -= 273.16
    ds = ds.load()
    # obs_types = ['drone', 'cam_temp', 'thermistor']
    # sim_cols = ['CONEVOL', 'TS', 'TBULK'] 
    # obs_cols = ['volume', 'surfTemp', 'bulkTemp'] 
    obs_types = ['drone', 'thermistor']
    sim_cols = ['CONEAREA', 'TS'] 
    obs_cols = ['area', 'bulkTemp'] 

    for i, obs in enumerate(obs_types):
        print(i, obs)
        df = pd.read_csv("../../data/input/" + icestupa + "/" + obs + ".csv")
        df['time'] = pd.to_datetime(df['TIMESTAMP'])
        df = df.set_index('time')
        if obs != 'drone':
            df = df[ time_start: time_end ]

        sim_data = (ds.sel(time=df.index)).to_dataframe()

        rmse = ((sim_data[sim_cols[i]].subtract(df[obs_cols[i]],axis=0))**2).mean()**.5
        corr = sim_data[sim_cols[i]].corr(df[obs_cols[i]])
        print(rmse, corr)

        # rmse_V = ((sim_data['CONEVOL'].subtract(df['volume'],axis=0))**2).mean()**.5
        # rmse_A = ((sim_data['CONEAREA'].subtract(df['volume'],axis=0))**2).mean()**.5
        # print(rmse_V, rmse_A)

    # dft = pd.read_csv("../../data/input/" + icestupa + "/thermistor.csv")
    # dft['time'] = pd.to_datetime(dft['TIMESTAMP'])
