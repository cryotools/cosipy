from cosipy.constants import Constants


def updateRoughness(GRID) -> float:
    """Update the surface roughness length.

    Implemented methods:

        - **Moelg12**: Linear increase in snow roughness length over
          time. From Mölg et al. (2009).

    Args:
        GRID (Grid): Glacier data structure.

    Returns:
        Updated surface roughness length [mm].
    """

    roughness_allowed = ["Moelg12"]
    if Constants.roughness_method == "Moelg12":
        sigma = method_Moelg(GRID)
    else:
        error_message = (
            f'Roughness method = "{Constants.roughness_method}" is not allowed,',
            f'must be one of {", ".join(roughness_allowed)}',
        )
        raise ValueError(" ".join(error_message))

    return sigma


def method_Moelg(GRID) -> float:
    """Update the roughness length.

    Adapted from Moelg et al. (2009), J.Clim. The roughness length of
    snow linearly increases from 0.24 (fresh snow) to 4 (firn) in
    60 days (1440 hours) i.e. (4-0.24)/1440 = 0.0026.

    Args:
        GRID (Grid): Glacier data structure.

    Returns:
        Surface roughness length, [mm]
    """

    # Get hours since the last snowfall
    # First get fresh snow properties (height and timestamp)
    _, fresh_snow_timestamp, _ = GRID.get_fresh_snow_props()

    # Get time difference between last snowfall and now
    hours_since_snowfall = fresh_snow_timestamp / 3600.0

    # Check whether snow or ice
    if GRID.get_node_density(0) <= Constants.snow_ice_threshold:
        sigma = min(
            Constants.roughness_fresh_snow
            + Constants.aging_factor_roughness * hours_since_snowfall,
            Constants.roughness_firn,
        )
    else:
        sigma = Constants.roughness_ice  # Roughness length, set to ice

    return sigma / 1000
