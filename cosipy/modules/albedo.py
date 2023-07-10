import numpy as np

import constants


def updateAlbedo(GRID) -> float:
    """This methods updates the albedo.

    Args:
        GRID (Grid): Glacier mesh.

    Returns:
        Updated surface albedo.
    """

    albedo_allowed = ["Oerlemans98", "Lejeune13"]
    if constants.albedo_method == "Oerlemans98":
        alphaMod = method_Oerlemans(GRID=GRID)
    elif constants.albedo_method == "Lejeune13":
        alphaMod = method_lejeune(GRID=GRID)  # snow-covered debris
    else:
        raise ValueError(
            f'Albedo method = "{constants.albedo_method}" is not allowed, must be one of {", ".join(albedo_allowed)}'
        )

    return alphaMod


def get_surface_properties(GRID) -> tuple:
    """Gets snowpack properties.

    Args:
        GRID (Grid): Glacier mesh.

    Returns:
        tuple[float, float, float]: Properties for height and timestamp of
        fresh snow, and hours elapsed since last snowfall.
    """

    # Get hours since the last snowfall
    # First get fresh snow properties (height and timestamp)
    fresh_snow_height, fresh_snow_timestamp, _ = GRID.get_fresh_snow_props()

    # Get time difference between last snowfall and now
    hours_since_snowfall = (fresh_snow_timestamp) / 3600.0

    # If fresh snow disappears faster than the snow ageing scale then set the hours_since_snowfall
    # to the old values of the underlying snowpack
    if (hours_since_snowfall < (constants.albedo_mod_snow_aging * 24)) & (
        fresh_snow_height <= 0.0  # possible bug? Negative snow height?
    ):
        GRID.set_fresh_snow_props_to_old_props()
        (
            fresh_snow_height,
            fresh_snow_timestamp,
            _,
        ) = GRID.get_fresh_snow_props()

        # Update time difference between last snowfall and now
        hours_since_snowfall = (fresh_snow_timestamp) / 3600.0

    return fresh_snow_height, fresh_snow_timestamp, hours_since_snowfall


def get_simple_albedo(elapsed_time: float) -> float:
    """Surface albedo neglecting snowpack depth (Oerlemans & Knap, 1998).

    Args:
        elapsed_time: Hours elapsed since last snowfall.

    Returns:
        Surface albedo without accounting for snowpack depth.
    """

    albedo = constants.albedo_firn + (
        constants.albedo_fresh_snow - constants.albedo_firn
    ) * np.exp((-elapsed_time) / (constants.albedo_mod_snow_aging * 24.0))

    return albedo


def method_Oerlemans(GRID):
    _, _, hours_since_snowfall = get_surface_properties(GRID)

    # Check if snow or ice
    if GRID.get_node_density(0) <= constants.snow_ice_threshold:
        # Get current snowheight from layer height
        h = GRID.get_total_snowheight()  # np.sum(GRID.get_height()[0:idx])

        # Surface albedo according to Oerlemans & Knap 1998, JGR)
        alphaSnow = get_simple_albedo(elapsed_time=hours_since_snowfall)
        alphaMod = alphaSnow + (constants.albedo_ice - alphaSnow) * np.exp(
            (-1.0 * h) / (constants.albedo_mod_snow_depth / 100.0)
        )

    else:
        # If no snow cover than set albedo to ice albedo
        alphaMod = constants.albedo_ice

    return alphaMod


# idea; albedo decay like (Brock et al. 2000)? or?
# Schmidt et al 2017 >doi:10.5194/tc-2017-67, 2017 use the same albedo
# parameterisation from Oerlemans and Knap 1998 with a slight updated
# implementation of considering the surface temperature?


def get_albedo_weight_lejeune(snow_depth: float) -> float:
    """Weighting for snow-covered debris albedo (Lejeune et al., 2007).

    Args:
        snow_depth: Height of snowpack above debris.

    Returns:
        Albedo weighting.
    """

    albedo_weight = min(
        1.0,
        (snow_depth / constants.critical_snowpack_thickness)
        ** constants.lejeune_weighting_coefficient,
    )

    return albedo_weight


def method_lejeune(GRID) -> float:
    """Snow-covered debris albedo (Lejeune et al., 2007).

    Args:
        GRID (Grid): Glacier mesh.

    Returns:
        Albedo for snow-covered debris.
    """

    fresh_snow_height, _, hours_since_snowfall = get_surface_properties(
        GRID=GRID
    )
    albedo_weight = get_albedo_weight_lejeune(snow_depth=fresh_snow_height)
    albedo_snow = get_simple_albedo(elapsed_time=hours_since_snowfall)

    albedo = (
        albedo_weight * albedo_snow
        + (1 - albedo_weight) * constants.albedo_debris
    )

    return albedo
