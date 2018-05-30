import numpy as np
import xarray as xr

from constants import *
from config import *
from core.grid import *

def init_snowpack(DATA):
    
    ''' INITIALIZATION '''
    
    # Initial snow height
    initial_snowheight = 0.2 
 
    # Init layers
    layer_heights =  np.ones(int(initial_snowheight // initial_snow_layer_heights)) * initial_snow_layer_heights
    layer_heights =  np.append(layer_heights, np.ones(int(initial_glacier_height // initial_glacier_layer_heights)) * initial_glacier_layer_heights)
    number_layers = len(layer_heights)

    # Init properties
    rho = ice_density * np.ones(len(layer_heights))
    temperature_profile = temperature_bottom * np.ones(len(layer_heights))
    liquid_water_content = np.zeros(number_layers)
    
    # Init density
    rho_top = 250.
    rho_bottom = 500.
    density_gradient = (rho_top-rho_bottom)/(initial_snowheight//initial_snow_layer_heights)
    for i in np.arange((initial_snowheight//initial_snow_layer_heights)):
       rho[int(i)] = rho_top - density_gradient * i 
    
    # Init temperature new
    temperature_gradient = (DATA.T2[0] - temperature_bottom) / (initial_glacier_height // initial_glacier_layer_heights)
    for i in np.arange(0 ,(initial_glacier_height // initial_glacier_layer_heights)):
        temperature_profile[int(i)] = DATA.T2[0] - temperature_gradient * i

    # Initialize grid, the grid class contains all relevant grid information
    GRID = Grid(layer_heights, rho, temperature_profile, liquid_water_content, debug_level)

    return GRID



def init_result_arrays(DATA):

    RESULT = xr.Dataset()
    RESULT.coords['lat'] = DATA.coords['lat']
    RESULT.coords['lon'] = DATA.coords['lon']
    RESULT.coords['time'] = DATA.coords['time']

    add_variable_1D(RESULT, np.empty_like(DATA.T2), 'SNOWHEIGHT', 'm', 'Snowheight')
    
    return RESULT

def add_variable_1D(ds, var, name, units, long_name):
    """ This function adds missing variables to the DATA class """
    ds[name] = xr.DataArray(var, coords=[('time', ds.time)])
    ds[name].attrs['units'] = units
    ds[name].attrs['long_name'] = long_name
    return ds

