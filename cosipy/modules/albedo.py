import numpy as np
from constants import albedo_method, albedo_fresh_snow, albedo_firn, albedo_ice, \
                      albedo_mod_snow_aging, albedo_mod_snow_depth, snow_ice_threshold
from cosipy.utils.options import read_opt

def updateAlbedo(GRID, opt_dict):
    """ This methods updates the albedo """
    # Read and set options
    read_opt(opt_dict, globals())
    albedo_allowed = ['Oerlemans98']
    if albedo_method == 'Oerlemans98':
        alphaMod = method_Oerlemans(GRID)

    else:
        raise ValueError("Albedo method = \"{:s}\" is not allowed, must be one of {:s}".format(albedo_method, ", ".join(albedo_allowed)))

    return alphaMod


def method_Oerlemans(GRID):

    # Get hours since the last snowfall
    # First get fresh snow properties (height and timestamp)
    fresh_snow_height, fresh_snow_timestamp, _  = GRID.get_fresh_snow_props()
    
    # Get time difference between last snowfall and now
    hours_since_snowfall = (fresh_snow_timestamp)/3600.0

    # If fresh snow disappears faster than the snow ageing scale then set the hours_since_snowfall
    # to the old values of the underlying snowpack
    if (hours_since_snowfall<(albedo_mod_snow_aging*24)) & (fresh_snow_height<0.0):
        GRID.set_fresh_snow_props_to_old_props()
        fresh_snow_height, fresh_snow_timestamp, _  = GRID.get_fresh_snow_props()
        
        # Update time difference between last snowfall and now
        hours_since_snowfall = (fresh_snow_timestamp)/3600.0

    # Check if snow or ice
    if (GRID.get_node_density(0) <= snow_ice_threshold):
        
        # Get current snowheight from layer height
        h = GRID.get_total_snowheight() #np.sum(GRID.get_height()[0:idx])

        # Surface albedo according to Oerlemans & Knap 1998, JGR)
        alphaSnow = albedo_firn + (albedo_fresh_snow - albedo_firn) *  np.exp((-hours_since_snowfall) / (albedo_mod_snow_aging * 24.0))
        alphaMod = alphaSnow + (albedo_ice - alphaSnow) *  np.exp((-1.0*h) / (albedo_mod_snow_depth / 100.0))

    else:
        # If no snow cover than set albedo to ice albedo
        alphaMod = albedo_ice

    return alphaMod

### idea; albedo decay like (Brock et al. 2000)? or?
### Schmidt et al 2017 >doi:10.5194/tc-2017-67, 2017 use the same albedo parameterisation from Oerlemans and Knap 1998 with a slight updated implementation of considering the surface temperature?
