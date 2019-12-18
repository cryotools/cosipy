from constants import *
import sys

def updateRoughness(GRID, timestamp):

    if roughness_method == 'Moelg12':
        sigma = method_Moelg(GRID, timestamp)
    else:
        print('Roughness parameterisation ', roughness_method, ' not available using default')
        sigma = method_Moelg(GRID, timestamp)

    return sigma


def method_Moelg(GRID, timestamp):

    """ This method updates the roughness length (Moelg et al 2009, J.Clim.)"""

    # Get hours since the last snowfall
    # First get fresh snow properties (height and timestamp)
    fresh_snow_height, fresh_snow_timestamp  = GRID.get_fresh_snow_props()

    # Get time difference between last snowfall and now
    hours_since_snowfall = (timestamp-fresh_snow_timestamp)/3600.0

    # Check whether snow or ice
    if (GRID.get_node_density(0) <= snow_ice_threshold):

        # Roughness length linear increase from 0.24 (fresh snow) to 4 (firn) in 60 days (1440 hours); (4-0.24)/1440 = 0.0026
        sigma = min(roughness_fresh_snow + aging_factor_roughness * hours_since_snowfall, roughness_firn)

    else:

        # Roughness length, set to ice
        sigma = roughness_ice

    return (sigma / 1000)
