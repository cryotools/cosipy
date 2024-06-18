from cosipy.constants import Constants


def updateRoughness(GRID):

    roughness_allowed = ['Moelg12']
    if Constants.roughness_method == 'Moelg12':
        sigma = method_Moelg(GRID)
    else:
        error_message = (
            f'Roughness method = "{Constants.roughness_method}" is not allowed,',
            f'must be one of {", ".join(roughness_allowed)}'
        )
        raise ValueError(" ".join(error_message))

    return sigma


def method_Moelg(GRID):
    """Update the roughness length (Moelg et al 2009, J.Clim.)."""

    # Get hours since the last snowfall
    # First get fresh snow properties (height and timestamp)
    fresh_snow_height, fresh_snow_timestamp, _  = GRID.get_fresh_snow_props()

    # Get time difference between last snowfall and now
    hours_since_snowfall = (fresh_snow_timestamp)/3600.0

    # Check whether snow or ice
    if (GRID.get_node_density(0) <= Constants.snow_ice_threshold):

        # Roughness length linear increase from 0.24 (fresh snow) to 4 (firn) in 60 days (1440 hours); (4-0.24)/1440 = 0.0026
        sigma = min(Constants.roughness_fresh_snow + Constants.aging_factor_roughness * hours_since_snowfall, Constants.roughness_firn)

    else:

        # Roughness length, set to ice
        sigma = Constants.roughness_ice

    return (sigma / 1000)
