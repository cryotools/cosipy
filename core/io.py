"""
 This file reads the input data (model forcing) and write the output to netcdf file
"""

import xarray as xr
import time
from config import input_netcdf, output_netcdf, plots, time_start, time_end
from postprocessing_modules.plots import plots_fluxes
import numpy as np

def preprocessing():
    """     
    PRES        ::   Air Pressure [hPa]
    N           ::   Cloud cover  [fraction][%/100]
    RH2         ::   Relative humidity (2m over ground)[%]
    SNOWFALL    ::   Snowfall per time step [m]
    G           ::   Solar radiation at each time step [W m-2]
    T2          ::   Air temperature (2m over ground) [K]
    U2          ::   Wind speed (magnitude) m/s
    """
    DATA = xr.open_dataset(input_netcdf)
    DATA['time'] = np.sort(DATA['time'].values)
    DATA = DATA.sel(time=slice(time_start, time_end))

    print('Checking input data .... \n')
    
    if ('T2' in DATA):
        print('Temperature data (T2) exists ')
    if ('RH2' in DATA):
        print('Relative humidity data (RH2) exists ')
    if ('G' in DATA):
        print('Shortwave data (G) exists ')
    if ('U2' in DATA):
        print('Wind velocity data (U2) exists ')
    if ('RRR' in DATA):
        print('Precipitation data (RRR) exists')
    if ('N' in DATA):
        print('Cloud cover data (N) exists ')
    if ('PRES' in DATA):
        print('Pressure data (PRES) exists ')

    return DATA

def add_variable(ds, var, name, units, long_name):
    """ This function adds missing variables to the DATA class """
    ds[name] = xr.DataArray(var, coords=[('time', ds.time)])
    ds[name].attrs['units'] = units
    ds[name].attrs['long_name'] = long_name

    return ds


