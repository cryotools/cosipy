import sys
import numpy as np
import xarray as xr
import logging
import math
from constants import *
from config import *
from cpkernel.grid import *

def init_snowpack(DATA):
    
    ''' INITIALIZATION '''

    logger = logging.getLogger(__name__)
    
    # Init layers
    if (initial_snowheight > 0.0):
        layer_heights =  np.ones(int(initial_snowheight // initial_snow_layer_heights)) * initial_snow_layer_heights
        layer_heights =  np.append(layer_heights, np.ones(int(initial_glacier_height // initial_glacier_layer_heights)) * initial_glacier_layer_heights)
    else:
        layer_heights = np.ones(int(initial_glacier_height // initial_glacier_layer_heights)) * initial_glacier_layer_heights

    number_layers = len(layer_heights)

    # Init properties
    layer_density = ice_density * np.ones(len(layer_heights))
    layer_T = temperature_bottom * np.ones(len(layer_heights))
    layer_LWC = np.zeros(number_layers)
    layer_cc = np.zeros(number_layers)
    layer_porosity = np.zeros(number_layers)
    layer_vol = np.zeros(number_layers)
    layer_refreeze = np.zeros(number_layers)

    if (initial_snowheight > 0.0):
        # Init density
        rho_top = initial_top_density_snowpack
        rho_bottom = initial_botton_density_snowpack
        density_gradient = (rho_top-rho_bottom)/(initial_snowheight//initial_snow_layer_heights)
        for i in np.arange((initial_snowheight//initial_snow_layer_heights)):
            layer_density[int(i)] = rho_top - density_gradient * i
    
    # Init temperature new
    temperature_gradient = (temperature_top - temperature_bottom) / (initial_glacier_height // initial_glacier_layer_heights)

    total_height = 0

    for i in np.arange(0 ,(initial_glacier_height // initial_glacier_layer_heights)):

        # Total height of overlying snowpack
        if int(i) > 0:
            total_height = total_height + layer_heights[int(i)]

        # Exponential decay of
        layer_T[int(i)] = float(temperature_bottom + (temperature_top - temperature_bottom) *  math.exp(const_init_temp * -total_height))

    # Initialize grid, the grid class contains all relevant grid information
    GRID = Grid(layer_heights, layer_density, layer_T, layer_LWC, layer_cc, layer_porosity, layer_vol, layer_refreeze, debug_level)

    return GRID



def load_snowpack(GRID_RESTART):
    """ Initialize grid from restart file """

    # Number of layers
    num_layers = np.int(GRID_RESTART.NLAYERS.values)
    
    # Init layer height
    layer_heights = GRID_RESTART.LAYER_HEIGHT[0:num_layers].values
    layer_density = GRID_RESTART.LAYER_RHO[0:num_layers].values
    layer_T = GRID_RESTART.LAYER_T[0:num_layers].values
    layer_LWC = GRID_RESTART.LAYER_LWC[0:num_layers].values
    layer_cc = GRID_RESTART.LAYER_CC[0:num_layers].values
    layer_porosity = GRID_RESTART.LAYER_POROSITY[0:num_layers].values
    layer_vol = GRID_RESTART.LAYER_VOL[0:num_layers].values
    layer_refreeze = GRID_RESTART.LAYER_REFREEZE[0:num_layers].values
    
    GRID = Grid(layer_heights, layer_density, layer_T, layer_LWC, layer_cc, layer_porosity, layer_vol, layer_refreeze, debug_level)

    if np.isnan(layer_T).any():
        GRID.grid_info()
        sys.exit(1) 
    
    return GRID












