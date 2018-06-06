"""
 This file reads the input data (model forcing) and write the output to netcdf file
"""

import xarray as xr
import numpy as np
import time

from config import input_netcdf, output_netcdf, plots, time_start, time_end

def read_data():
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


def init_result_dataset(DATA):
    """ This function creates the result dataset 
    Args:
        
        DATA    ::  DATA structure 
        
    Returns:
        
        RESULT  ::  one-dimensional RESULT structure"""

    RESULT = xr.Dataset()
    RESULT.coords['lat'] = DATA.coords['lat']
    RESULT.coords['lon'] = DATA.coords['lon']
    RESULT.coords['time'] = DATA.coords['time']

    add_variable_2D(RESULT, np.full_like(DATA.T2, -999, dtype=np.double), 'SNOWHEIGHT', 'm', 'snowheight')
    add_variable_2D(RESULT, np.full_like(DATA.T2, -999, dtype=np.double), 'EVAPORATION', 'm w.e.q.', 'evaporation')
    add_variable_2D(RESULT, np.full_like(DATA.T2, -999, dtype=np.double), 'SUBLIMATION', 'm w.e.q.', 'sublimation')
    add_variable_2D(RESULT, np.full_like(DATA.T2, -999, dtype=np.double), 'MELT', 'm w.e.q.', 'surface melt')
    add_variable_2D(RESULT, np.full_like(DATA.T2, -999, dtype=np.double), 'H', 'W m^-2', 'sensible heat flux')
    add_variable_2D(RESULT, np.full_like(DATA.T2, -999, dtype=np.double), 'LE', 'W m^-2', 'latent heat flux')
    add_variable_2D(RESULT, np.full_like(DATA.T2, -999, dtype=np.double), 'B', 'W m^-2', 'ground heat flux')
    add_variable_2D(RESULT, np.full_like(DATA.T2, -999, dtype=np.double), 'TS', 'K', 'surface temperature')
    add_variable_2D(RESULT, np.full_like(DATA.T2, -999, dtype=np.double), 'ALB', '-', 'surface albedo')
    add_variable_2D(RESULT, np.full_like(DATA.T2, -999, dtype=np.double), 'DEPO', 'm w.e.q.', 'deposition')
    add_variable_2D(RESULT, np.full_like(DATA.T2, -999, dtype=np.double), 'CONDEN', 'm w.e.q.', 'condensation')
    add_variable_2D(RESULT, np.full_like(DATA.T2, -999, dtype=np.double), 'LWout', 'W m^-2', 'longwave incoming radiation')
    add_variable_2D(RESULT, np.full_like(DATA.T2, -999, dtype=np.double), 'LWin', 'W m^-2', 'longwave outgoing radiation')
    add_variable_2D(RESULT, np.full_like(DATA.T2, -999, dtype=np.double), 'MB', 'm w.e.q.', 'mass balance')
    add_variable_2D(RESULT, np.full_like(DATA.T2, -999, dtype=np.double), 'MH', 'm w.e.q.', 'melt height')
    add_variable_2D(RESULT, np.full_like(DATA.T2, -999, dtype=np.double), 'NL', '-', 'number layer')
    add_variable_2D(RESULT, np.full_like(DATA.T2, -999, dtype=np.double), 'RF', 'm w.e.q.', 'refreezing')
    add_variable_2D(RESULT, np.full_like(DATA.T2, -999, dtype=np.double), 'Q', 'm w.e.q.', 'runoff')
    add_variable_2D(RESULT, np.full_like(DATA.T2, -999, dtype=np.double), 'SM', 'm w.e.q.', 'surface melt')

    return RESULT


def init_result_dataset_1D(DATA):
    """ This function creates the 1D result dataset 
    Args:
        
        DATA    ::  DATA structure 
        
    Returns:
        
        RESULT  ::  one-dimensional RESULT structure"""

    RESULT = xr.Dataset()
    RESULT.coords['lat'] = DATA.coords['lat']
    RESULT.coords['lon'] = DATA.coords['lon']
    RESULT.coords['time'] = DATA.coords['time']

    add_variable_1D(RESULT, np.full_like(DATA.T2, -999, dtype=np.double), 'SNOWHEIGHT', 'm', 'snowheight')
    add_variable_1D(RESULT, np.full_like(DATA.T2, -999, dtype=np.double), 'EVAPORATION', 'm w.e.q.', 'evaporation')
    add_variable_1D(RESULT, np.full_like(DATA.T2, -999, dtype=np.double), 'SUBLIMATION', 'm w.e.q.', 'sublimation')
    add_variable_1D(RESULT, np.full_like(DATA.T2, -999, dtype=np.double), 'MELT', 'm w.e.q.', 'surface melt')
    add_variable_1D(RESULT, np.full_like(DATA.T2, -999, dtype=np.double), 'H', 'W m^-2', 'sensible heat flux')
    add_variable_1D(RESULT, np.full_like(DATA.T2, -999, dtype=np.double), 'LE', 'W m^-2', 'latent heat flux')
    add_variable_1D(RESULT, np.full_like(DATA.T2, -999, dtype=np.double), 'B', 'W m^-2', 'ground heat flux')
    add_variable_1D(RESULT, np.full_like(DATA.T2, -999, dtype=np.double), 'TS', 'K', 'surface temperature')
    add_variable_1D(RESULT, np.full_like(DATA.T2, -999, dtype=np.double), 'ALB', '-', 'surface albedo')
    add_variable_1D(RESULT, np.full_like(DATA.T2, -999, dtype=np.double), 'DEPO', 'm w.e.q.', 'deposition')
    add_variable_1D(RESULT, np.full_like(DATA.T2, -999, dtype=np.double), 'CONDEN', 'm w.e.q.', 'condensation')
    add_variable_1D(RESULT, np.full_like(DATA.T2, -999, dtype=np.double), 'LWout', 'W m^-2', 'longwave incoming radiation')
    add_variable_1D(RESULT, np.full_like(DATA.T2, -999, dtype=np.double), 'LWin', 'W m^-2', 'longwave outgoing radiation')
    add_variable_1D(RESULT, np.full_like(DATA.T2, -999, dtype=np.double), 'MB', 'm w.e.q.', 'mass balance')
    add_variable_1D(RESULT, np.full_like(DATA.T2, -999, dtype=np.double), 'MH', 'm w.e.q.', 'melt heigt')
    add_variable_1D(RESULT, np.full_like(DATA.T2, -999, dtype=np.double), 'NL', 'm w.e.q.', 'number layer')
    add_variable_1D(RESULT, np.full_like(DATA.T2, -999, dtype=np.double), 'RF', 'm w.e.q.', 'refreezing')
    add_variable_1D(RESULT, np.full_like(DATA.T2, -999, dtype=np.double), 'Q', 'm w.e.q.', 'runoff')
    add_variable_1D(RESULT, np.full_like(DATA.T2, -999, dtype=np.double), 'SM', 'm w.e.q.', 'surface melt')

    return RESULT

def add_variable_1D(ds, var, name, units, long_name):
    """ This function adds missing variables to the DATA class """
    ds[name] = xr.DataArray(var, coords=[('time', ds.time)])
    ds[name].attrs['units'] = units
    ds[name].attrs['long_name'] = long_name
    return ds

def add_variable_2D(ds, var, name, units, long_name):
    """ This function adds missing variables to the DATA class """
    ds[name] = (('lat','lon','time'), var)
    ds[name].attrs['units'] = units
    ds[name].attrs['long_name'] = long_name
    return ds

