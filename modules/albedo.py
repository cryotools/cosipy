import numpy as np
from constants import *


def updateAlbedo(GRID, evdiff):
    """ This methods updates the albedo """

    # Check if snow or ice
    if (GRID.get_node_density(0) <= snow_ice_threshold):
    
        # Get current snowheight from layer height 
        idx = (next((i for i, x in enumerate(GRID.get_density()) if x >= snow_ice_threshold), None))
        h = np.sum(GRID.get_height()[0:idx])
    
        # Surface albedo according to Oerlemans & Knap 1998, JGR)
        alphaSnow = albedo_firn + (albedo_fresh_snow - albedo_firn) * \
                                  np.exp((-evdiff) / (albedo_mod_snow_aging * 24.0))
        alphaMod = alphaSnow + (albedo_ice - alphaSnow) * \
                               np.exp((-1.0*h) / (albedo_mod_snow_depth / 100.0))
    
    else:
    
        # If no snow cover than set albedo to ice albedo
        alphaMod = albedo_ice


    return alphaMod


### idea; have a deeper look if it would by worthwhile; albedo decay like (Brock et al. 2000)
### Schmidt et al 2017 >doi:10.5194/tc-2017-67, 2017 use the same albedo parameterisation from Oerlemans and Knap 1998 with a slight updated implementation of considering the surface temperature?
