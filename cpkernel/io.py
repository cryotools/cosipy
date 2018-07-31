"""
 This file reads the input data (model forcing) and write the output to netcdf file
"""

import sys
import xarray as xr
import pandas as pd
import numpy as np
import time

from config import * 
from modules.radCor import correctRadiation


def read_data(restart_date=None):
    """     
    PRES        ::   Air Pressure [hPa]
    N           ::   Cloud cover  [fraction][%/100]
    RH2         ::   Relative humidity (2m over ground)[%]
    RRR         ::   Precipitation per time step [m]
    G           ::   Solar radiation at each time step [W m-2]
    T2          ::   Air temperature (2m over ground) [K]
    U2          ::   Wind speed (magnitude) m/s
    """

    # Open input dataset
    DATA = xr.open_dataset(input_netcdf)
    DATA['time'] = np.sort(DATA['time'].values)
    
    # Check if restart
    if restart_date is None:
        print('--------------------------------------------------------------')
        print('\t Integration from %s to %s' % (time_start, time_end))
        print('--------------------------------------------------------------\n')
        DATA = DATA.sel(time=slice(time_start, time_end))   # Select dates from config.py
    else:
        # Get end date from the input data
        end_date = DATA.time[-1]

        # There is nothing to do if the dates are equal
        if (restart_date==end_date):
            print('Start date equals end date ... no new data ... EXIT')
            sys.exit(1)
        else:
            # otherwise, run the model from the restart date to the end of the input data
            print('Starting from %s (from restart file) to %s (from config.py) \n' % (restart_date.values, end_date.values))
            DATA = DATA.sel(time=slice(restart_date, end_date))

    print('--------------------------------------------------------------')
    print('Checking input data .... \n')
    
    if ('T2' in DATA):
        print('Temperature data (T2) ... ok ')
    if ('RH2' in DATA):
        print('Relative humidity data (RH2) ... ok ')
    if ('G' in DATA):
        print('Shortwave data (G) ... ok ')
    if ('U2' in DATA):
        print('Wind velocity data (U2) ... ok ')
    if ('RRR' in DATA):
        print('Precipitation data (RRR) ... ok ')
    if ('N' in DATA):
        print('Cloud cover data (N) ... ok ')
    if ('PRES' in DATA):
        print('Pressure data (PRES) ... ok ')
    if ('LWin' in DATA):
        print('Incoming longwave data (LWin) ... ok ')
    

    return DATA



def init_result_dataset(DATA):
    """ This function creates the result file 
    Args:
        
        DATA    ::  DATA structure 
        
    Returns:
        
        RESULT  ::  one-dimensional RESULT structure"""

    RESULT = xr.Dataset()
    RESULT.coords['lat'] = DATA.coords['lat']
    RESULT.coords['lon'] = DATA.coords['lon']
    RESULT.coords['time'] = DATA.coords['time']
    RESULT.coords['layer'] = np.arange(max_layers)

    add_variable_along_latlontime(RESULT, np.full_like(DATA.T2, np.nan), 'SNOWHEIGHT', 'm', 'Snowheight')
    add_variable_along_latlontime(RESULT, np.full_like(DATA.T2, np.nan), 'EVAPORATION', 'm w.e.q.', 'Evaporation')
    add_variable_along_latlontime(RESULT, np.full_like(DATA.T2, np.nan), 'SUBLIMATION', 'm w.e.q.', 'Sublimation')
    add_variable_along_latlontime(RESULT, np.full_like(DATA.T2, np.nan), 'MELT', 'm w.e.q.', 'Surface melt')
    add_variable_along_latlontime(RESULT, np.full_like(DATA.T2, np.nan), 'LWin', 'W m^-2', 'Incoming longwave radiation')
    add_variable_along_latlontime(RESULT, np.full_like(DATA.T2, np.nan), 'LWout', 'W m^-2', 'Outgoing longwave radiation')
    add_variable_along_latlontime(RESULT, np.full_like(DATA.T2, np.nan), 'H', 'W m^-2', 'Sensible heat flux')
    add_variable_along_latlontime(RESULT, np.full_like(DATA.T2, np.nan), 'LE', 'W m^-2', 'Latent heat flux')
    add_variable_along_latlontime(RESULT, np.full_like(DATA.T2, np.nan), 'B', 'W m^-2', 'Ground heat flux')
    add_variable_along_latlontime(RESULT, np.full_like(DATA.T2, np.nan), 'TS', 'K', 'Surface temperature')
    add_variable_along_latlontime(RESULT, np.full_like(DATA.T2, np.nan), 'RH2', '%', 'Relative humidity at 2 m')
    add_variable_along_latlontime(RESULT, np.full_like(DATA.T2, np.nan), 'T2', 'K', 'Air temperature at 2 m')
    add_variable_along_latlontime(RESULT, np.full_like(DATA.T2, np.nan), 'G', 'W m^-2', 'Incoming shortwave radiation')
    add_variable_along_latlontime(RESULT, np.full_like(DATA.T2, np.nan), 'U2', 'm s^-1', 'Wind velocity at 2 m')
    add_variable_along_latlontime(RESULT, np.full_like(DATA.T2, np.nan), 'N', '-', 'Cloud fraction')
    add_variable_along_latlontime(RESULT, np.full_like(DATA.T2, np.nan), 'Z0', 'm', 'Roughness length')
    add_variable_along_latlontime(RESULT, np.full_like(DATA.T2, np.nan), 'ALBEDO', '-', 'Albedo')
    
    add_variable_along_latlontime(RESULT, np.full_like(DATA.T2, np.nan), 'NLAYERS', '-', 'Number of layers')
    
    add_variable_along_latlonlayertime(RESULT, np.full((DATA.T2.shape[0], DATA.T2.shape[1], RESULT.coords['layer'].shape[0],RESULT.coords['time'].shape[0]), 
                                                        np.nan), 'LAYER_HEIGHT', 'm', 'Height of each layer')
    add_variable_along_latlonlayertime(RESULT, np.full((DATA.T2.shape[0], DATA.T2.shape[1], RESULT.coords['layer'].shape[0],RESULT.coords['time'].shape[0]), 
                                                        np.nan), 'LAYER_RHO', 'm', 'Layer density')
    add_variable_along_latlonlayertime(RESULT, np.full((DATA.T2.shape[0], DATA.T2.shape[1], RESULT.coords['layer'].shape[0],RESULT.coords['time'].shape[0]), 
                                                        np.nan), 'LAYER_T', 'm', 'Layer temperature')
    add_variable_along_latlonlayertime(RESULT, np.full((DATA.T2.shape[0], DATA.T2.shape[1], RESULT.coords['layer'].shape[0],RESULT.coords['time'].shape[0]), 
                                                        np.nan), 'LAYER_LWC', 'm', 'Liquid water content of layer')
    add_variable_along_latlonlayertime(RESULT, np.full((DATA.T2.shape[0], DATA.T2.shape[1], RESULT.coords['layer'].shape[0],RESULT.coords['time'].shape[0]), 
                                                        np.nan), 'LAYER_CC', 'm', 'Cold content of each layer')
    add_variable_along_latlonlayertime(RESULT, np.full((DATA.T2.shape[0], DATA.T2.shape[1], RESULT.coords['layer'].shape[0],RESULT.coords['time'].shape[0]), 
                                                        np.nan), 'LAYER_POROSITY', 'm', 'Porosity of each layer')
    add_variable_along_latlonlayertime(RESULT, np.full((DATA.T2.shape[0], DATA.T2.shape[1], RESULT.coords['layer'].shape[0],RESULT.coords['time'].shape[0]), 
                                                        np.nan), 'LAYER_VOL', 'm', 'Volumetic ice content of each layer')
    print('\n') 
    print('Output dataset ... ok \n')
    print('--------------------------------------------------------------\n')

    return RESULT



def init_result_dataset_point(DATA, max_layers):
    """ This function creates the result dataset for a grid point 
    Args:
        
        DATA    ::  DATA structure 
        
    Returns:
        
        RESULT  ::  one-dimensional RESULT structure"""

    RESULT = xr.Dataset()
    RESULT.coords['lat'] = DATA.coords['lat']
    RESULT.coords['lon'] = DATA.coords['lon']
    RESULT.coords['time'] = DATA.coords['time']
    RESULT.coords['layer'] = np.arange(max_layers)

    add_variable_along_time(RESULT, np.full_like(DATA.T2, np.nan), 'SNOWHEIGHT', 'm', 'Snowheight')
    add_variable_along_time(RESULT, np.full_like(DATA.T2, np.nan), 'EVAPORATION', 'm w.e.q.', 'Evaporation')
    add_variable_along_time(RESULT, np.full_like(DATA.T2, np.nan), 'SUBLIMATION', 'm w.e.q.', 'Sublimation')
    add_variable_along_time(RESULT, np.full_like(DATA.T2, np.nan), 'MELT', 'm w.e.q.', 'Surface melt')
    add_variable_along_time(RESULT, np.full_like(DATA.T2, np.nan), 'LWin', 'W m^-2', 'Incoming longwave radiation')
    add_variable_along_time(RESULT, np.full_like(DATA.T2, np.nan), 'LWout', 'W m^-2', 'Outgoing longwave radiation')
    add_variable_along_time(RESULT, np.full_like(DATA.T2, np.nan), 'H', 'W m^-2', 'Sensible heat flux')
    add_variable_along_time(RESULT, np.full_like(DATA.T2, np.nan), 'LE', 'W m^-2', 'Latent heat flux')
    add_variable_along_time(RESULT, np.full_like(DATA.T2, np.nan), 'B', 'W m^-2', 'Ground heat flux')
    add_variable_along_time(RESULT, np.full_like(DATA.T2, np.nan), 'TS', 'K', 'Surface temperature')
    add_variable_along_time(RESULT, np.full_like(DATA.T2, np.nan), 'RH2', '%', 'Relative humidity at 2 m')
    add_variable_along_time(RESULT, np.full_like(DATA.T2, np.nan), 'T2', 'K', 'Air temperature at 2 m')
    add_variable_along_time(RESULT, np.full_like(DATA.T2, np.nan), 'G', 'W m^-2', 'Incoming shortwave radiation')
    add_variable_along_time(RESULT, np.full_like(DATA.T2, np.nan), 'U2', 'm s^-1', 'Wind velocity at 2 m')
    add_variable_along_time(RESULT, np.full_like(DATA.T2, np.nan), 'N', '-', 'Cloud fraction')
    add_variable_along_time(RESULT, np.full_like(DATA.T2, np.nan), 'Z0', 'm', 'Roughness length')
    add_variable_along_time(RESULT, np.full_like(DATA.T2, np.nan), 'ALBEDO', '-', 'Albedo')
    
    add_variable_along_time(RESULT, np.full_like(DATA.T2, np.nan), 'NLAYERS', '-', 'Number of layers')
    add_variable_along_layertime(RESULT, np.full((RESULT.coords['layer'].shape[0], RESULT.coords['time'].shape[0]), np.nan), 'LAYER_HEIGHT', 'm', 'Layer height')
    add_variable_along_layertime(RESULT, np.full((RESULT.coords['layer'].shape[0], RESULT.coords['time'].shape[0]), np.nan), 'LAYER_RHO', 'm', 'Density of layer')
    add_variable_along_layertime(RESULT, np.full((RESULT.coords['layer'].shape[0], RESULT.coords['time'].shape[0]), np.nan), 'LAYER_T', 'm', 'Layer temperature')
    add_variable_along_layertime(RESULT, np.full((RESULT.coords['layer'].shape[0], RESULT.coords['time'].shape[0]), np.nan), 'LAYER_LWC', 'm', 'LWC of each layer')
    add_variable_along_layertime(RESULT, np.full((RESULT.coords['layer'].shape[0], RESULT.coords['time'].shape[0]), np.nan), 'LAYER_CC', 'm', 'Cold content of each layer')
    add_variable_along_layertime(RESULT, np.full((RESULT.coords['layer'].shape[0], RESULT.coords['time'].shape[0]), np.nan), 'LAYER_POROSITY', 'm', 'Porosity of each layer')
    add_variable_along_layertime(RESULT, np.full((RESULT.coords['layer'].shape[0], RESULT.coords['time'].shape[0]), np.nan), 'LAYER_VOL', 'm', 'Volumetric ice content of each layer')
    
    return RESULT



def add_variable_along_time(ds, var, name, units, long_name):
    """ This function adds missing variables to the DATA class """
    ds[name] = xr.DataArray(var, coords=[('time', ds.time)])
    ds[name].attrs['units'] = units
    ds[name].attrs['long_name'] = long_name
    #ds[name].encoding['dtype'] = 'int16'
    #ds[name].encoding['scale_factor'] =  0.01
    ds[name].encoding['_FillValue'] = -9999
    #ds[name].attrs['missing_value'] = -9999
    return ds

def add_variable_along_latlontime(ds, var, name, units, long_name):
    """ This function adds missing variables to the DATA class """
    ds[name] = (('lat','lon','time'), var)
    ds[name].attrs['units'] = units
    ds[name].attrs['long_name'] = long_name
    #ds[name].encoding['dtype'] = 'int16'
    #ds[name].encoding['scale_factor'] =  0.01
    ds[name].encoding['_FillValue'] = -9999
    #ds[name].attrs['missing_value'] = -9999
    return ds

def add_variable_along_latlonlayertime(ds, var, name, units, long_name):
    """ This function adds missing variables to the DATA class """
    ds[name] = (('lat','lon','layer','time'), var)
    ds[name].attrs['units'] = units
    ds[name].attrs['long_name'] = long_name
    #ds[name].encoding['dtype'] = 'int16'
    #ds[name].encoding['scale_factor'] =  0.01
    ds[name].encoding['_FillValue'] = -9999
    #ds[name].attrs['missing_value'] = -9999
    return ds

def add_variable_along_layertime(ds, var, name, units, long_name):
    """ This function adds missing variables to the DATA class """
    ds[name] = (('layer','time'), var)
    ds[name].attrs['units'] = units
    ds[name].attrs['long_name'] = long_name
    #ds[name].encoding['dtype'] = 'int16'
    #ds[name].encoding['scale_factor'] =  0.01
    ds[name].encoding['_FillValue'] = -9999
    #ds[name].attrs['missing_value'] = -9999
    return ds
