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

    add_variable_2D(RESULT, np.full_like(DATA.T2, np.nan, dtype=np.double), 'SWin', 'W m\u207B\xB2', 'incoming shortwave radiation')
    add_variable_2D(RESULT, np.full_like(DATA.T2, np.nan, dtype=np.double), 'SWnet', 'W m\u207B\xB2', 'net shortwave radiation')
    add_variable_2D(RESULT, np.full_like(DATA.T2, np.nan, dtype=np.double), 'LWin', 'W m\u207B\xB2', 'incoming longwave radiation')
    add_variable_2D(RESULT, np.full_like(DATA.T2, np.nan, dtype=np.double), 'LWout', 'W m\u207B\xB2', 'outgoing longwave radiation')
    add_variable_2D(RESULT, np.full_like(DATA.T2, np.nan, dtype=np.double), 'H', 'W m\u207B\xB2', 'sensible heat flux')
    add_variable_2D(RESULT, np.full_like(DATA.T2, np.nan, dtype=np.double), 'LE', 'W m\u207B\xB2', 'latent heat flux')
    add_variable_2D(RESULT, np.full_like(DATA.T2, np.nan, dtype=np.double), 'B', 'W m\u207B\xB2', 'ground heat flux')
    add_variable_2D(RESULT, np.full_like(DATA.T2, np.nan, dtype=np.double), 'ME', 'W m\u207B\xB2', 'melt energy')
    add_variable_2D(RESULT, np.full_like(DATA.T2, np.nan, dtype=np.double), 'MB', 'm w.e.', 'mass balance')
    add_variable_2D(RESULT, np.full_like(DATA.T2, np.nan, dtype=np.double), 'surfMB', 'm w.e.', 'surface mass balance')
    add_variable_2D(RESULT, np.full_like(DATA.T2, np.nan, dtype=np.double), 'intMB', 'm w.e.', 'internal mass balance')
    add_variable_2D(RESULT, np.full_like(DATA.T2, np.nan, dtype=np.double), 'SUBLI', 'm w.e.', 'sublimation')
    add_variable_2D(RESULT, np.full_like(DATA.T2, np.nan, dtype=np.double), 'DEPO', 'm w.e.', 'deposition')
    add_variable_2D(RESULT, np.full_like(DATA.T2, np.nan, dtype=np.double), 'EVAPO', 'm w.e.', 'evaporation')
    add_variable_2D(RESULT, np.full_like(DATA.T2, np.nan, dtype=np.double), 'CONDEN', 'm w.e.', 'condensation')
    add_variable_2D(RESULT, np.full_like(DATA.T2, np.nan, dtype=np.double), 'surfM', 'm w.e.', 'surface melt')
    add_variable_2D(RESULT, np.full_like(DATA.T2, np.nan, dtype=np.double), 'subM', 'm w.e.', 'subsurface melt')
    add_variable_2D(RESULT, np.full_like(DATA.T2, np.nan, dtype=np.double), 'Q', 'm w.e.', 'runoff')
    add_variable_2D(RESULT, np.full_like(DATA.T2, np.nan, dtype=np.double), 'RF', 'm w.e.', 'refreezing')
    add_variable_2D(RESULT, np.full_like(DATA.T2, np.nan, dtype=np.double), 'SF', 'm w.e.', 'snowfall')
    add_variable_2D(RESULT, np.full_like(DATA.T2, np.nan, dtype=np.double), 'surfMH', 'm w.e.', 'surface melt height')
    add_variable_2D(RESULT, np.full_like(DATA.T2, np.nan, dtype=np.double), 'SH', 'm', 'snowheight')
    add_variable_2D(RESULT, np.full_like(DATA.T2, np.nan, dtype=np.double), 'surfT', 'K', 'surface temperature')
    add_variable_2D(RESULT, np.full_like(DATA.T2, np.nan, dtype=np.double), 'surfA', '-', 'surface albedo')
    add_variable_2D(RESULT, np.full_like(DATA.T2, np.nan, dtype=np.double), 'NL', '-', 'number layer')

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

    add_variable_1D(RESULT, np.full_like(DATA.T2, np.nan, dtype=np.double), 'SWin', 'W m\u207B\xB2', 'incoming shortwave radiation')
    add_variable_1D(RESULT, np.full_like(DATA.T2, np.nan, dtype=np.double), 'SWnet', 'W m\u207B\xB2', 'net shortwave radiation')
    add_variable_1D(RESULT, np.full_like(DATA.T2, np.nan, dtype=np.double), 'LWin', 'W m\u207B\xB2', 'incoming longwave radiation')
    add_variable_1D(RESULT, np.full_like(DATA.T2, np.nan, dtype=np.double), 'LWout', 'W m\u207B\xB2', 'outgoing longwave radiation')
    add_variable_1D(RESULT, np.full_like(DATA.T2, np.nan, dtype=np.double), 'H', 'W m\u207B\xB2', 'sensible heat flux')
    add_variable_1D(RESULT, np.full_like(DATA.T2, np.nan, dtype=np.double), 'LE', 'W m\u207B\xB2', 'latent heat flux')
    add_variable_1D(RESULT, np.full_like(DATA.T2, np.nan, dtype=np.double), 'B', 'W m\u207B\xB2', 'ground heat flux')
    add_variable_1D(RESULT, np.full_like(DATA.T2, np.nan, dtype=np.double), 'ME', 'W m\u207B\xB2', 'melt energy')
    add_variable_1D(RESULT, np.full_like(DATA.T2, np.nan, dtype=np.double), 'MB', 'm w.e.', 'mass balance')
    add_variable_1D(RESULT, np.full_like(DATA.T2, np.nan, dtype=np.double), 'surfMB', 'm w.e.', 'surface mass balance')
    add_variable_1D(RESULT, np.full_like(DATA.T2, np.nan, dtype=np.double), 'intMB', 'm w.e.', 'internal mass balance')
    add_variable_1D(RESULT, np.full_like(DATA.T2, np.nan, dtype=np.double), 'SUBLI', 'm w.e.', 'sublimation')
    add_variable_1D(RESULT, np.full_like(DATA.T2, np.nan, dtype=np.double), 'DEPO', 'm w.e.', 'deposition')
    add_variable_1D(RESULT, np.full_like(DATA.T2, np.nan, dtype=np.double), 'EVAPO', 'm w.e.', 'evaporation')
    add_variable_1D(RESULT, np.full_like(DATA.T2, np.nan, dtype=np.double), 'CONDEN', 'm w.e.', 'condensation')
    add_variable_1D(RESULT, np.full_like(DATA.T2, np.nan, dtype=np.double), 'surfM', 'm w.e.', 'surface melt')
    add_variable_1D(RESULT, np.full_like(DATA.T2, np.nan, dtype=np.double), 'subM', 'm w.e.', 'subsurface melt')
    add_variable_1D(RESULT, np.full_like(DATA.T2, np.nan, dtype=np.double), 'Q', 'm w.e.', 'runoff')
    add_variable_1D(RESULT, np.full_like(DATA.T2, np.nan, dtype=np.double), 'RF', 'm w.e.', 'refreezing')
    add_variable_1D(RESULT, np.full_like(DATA.T2, np.nan, dtype=np.double), 'SF', 'm w.e.', 'snowfall')
    add_variable_1D(RESULT, np.full_like(DATA.T2, np.nan, dtype=np.double), 'surfMH', 'm w.e.', 'surface melt height')
    add_variable_1D(RESULT, np.full_like(DATA.T2, np.nan, dtype=np.double), 'SH', 'm', 'snowheight')
    add_variable_1D(RESULT, np.full_like(DATA.T2, np.nan, dtype=np.double), 'surfT', 'K', 'surface temperature')
    add_variable_1D(RESULT, np.full_like(DATA.T2, np.nan, dtype=np.double), 'surfA', '-', 'surface albedo')
    add_variable_1D(RESULT, np.full_like(DATA.T2, np.nan, dtype=np.double), 'NL', '-', 'number layer')

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