from constants import roughness_method, roughness_fresh_snow, \
                      roughness_firn, roughness_ice, snow_ice_threshold, \
                      aging_factor_roughness
from cosipy.utils.options import read_opt

def updateRoughness(GRID, opt_dict=None):

    # Read and set options
    read_opt(opt_dict, globals())

    roughness_allowed = ['Moelg12']
    if roughness_method == 'Moelg12':
        sigma = method_Moelg(GRID)
    else:
        raise ValueError("Roghness method = \"{:s}\" is not allowed, must be one of {:s}".format(roughness_method,", ".join(roughness_allowed)))

    return sigma


def method_Moelg(GRID):

    """ This method updates the roughness length (Moelg et al 2009, J.Clim.)"""

    # Get hours since the last snowfall
    # First get fresh snow properties (height and timestamp)
    fresh_snow_height, fresh_snow_timestamp, _  = GRID.get_fresh_snow_props()

    # Get time difference between last snowfall and now
    hours_since_snowfall = (fresh_snow_timestamp)/3600.0

    # Check whether snow or ice
    if (GRID.get_node_density(0) <= snow_ice_threshold):

        # Roughness length linear increase from 0.24 (fresh snow) to 4 (firn) in 60 days (1440 hours); (4-0.24)/1440 = 0.0026
        sigma = min(roughness_fresh_snow + aging_factor_roughness * hours_since_snowfall, roughness_firn)

    else:

        # Roughness length, set to ice
        sigma = roughness_ice

    return (sigma / 1000)
