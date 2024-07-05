import numpy as np

import constants


def updateAlbedo(GRID, surface_temperature, albedo_snow):
    """Updates the albedo."""
    albedo_allowed = ["Oerlemans98", "Bougamont05"]
    if constants.albedo_method == "Oerlemans98":
        alphaMod = method_Oerlemans(GRID)
    elif constants.albedo_method == "Bougamont05":
        alphaMod, albedo_snow = method_Bougamont(GRID, surface_temperature, albedo_snow)
    else:
        error_message = (
            f'Albedo method = "{constants.albedo_method}"',
            f"is not allowed, must be one of",
            f'{", ".join(albedo_allowed)}',
        )
        raise ValueError(" ".join(error_message))

    return alphaMod, albedo_snow


def method_Oerlemans(GRID):
    # Get hours since the last snowfall
    # First get fresh snow properties (height and timestamp)
    fresh_snow_height, fresh_snow_timestamp, _ = GRID.get_fresh_snow_props()

    # Get time difference between last snowfall and now
    hours_since_snowfall = (fresh_snow_timestamp) / 3600.0

    # If fresh snow disappears faster than the snow ageing scale then set the hours_since_snowfall
    # to the old values of the underlying snowpack
    if (hours_since_snowfall < (constants.albedo_mod_snow_aging * 24)) & (
        fresh_snow_height <= 0.0
    ):
        GRID.set_fresh_snow_props_to_old_props()
        fresh_snow_height, fresh_snow_timestamp, _ = GRID.get_fresh_snow_props()

        # Update time difference between last snowfall and now
        hours_since_snowfall = (fresh_snow_timestamp) / 3600.0

    # Check if snow or ice
    if GRID.get_node_density(0) <= constants.snow_ice_threshold:
        # Get current snowheight from layer height
        h = GRID.get_total_snowheight()  # np.sum(GRID.get_height()[0:idx])

        # Surface albedo according to Oerlemans & Knap 1998, JGR)
        alphaSnow = constants.albedo_firn + (
            constants.albedo_fresh_snow - constants.albedo_firn
        ) * np.exp((-hours_since_snowfall) / (constants.albedo_mod_snow_aging * 24.0))
        alphaMod = alphaSnow + (constants.albedo_ice - alphaSnow) * np.exp(
            (-1.0 * h) / (constants.albedo_mod_snow_depth / 100.0)
        )

    else:
        # If no snow cover than set albedo to ice albedo
        alphaMod = constants.albedo_ice

    return alphaMod


def method_Bougamont(GRID, surface_temperature, albedo_snow):
    # Get hours since the last snowfall
    # First get fresh snow properties (height and timestamp)
    _, fresh_snow_timestamp, _ = GRID.get_fresh_snow_props()

    # Get time difference between last snowfall and now:
    hours_since_snowfall = (fresh_snow_timestamp) / 3600.0

    # Convert integration time from seconds to days:
    dt_days = constants.dt / 86400.0

    # Note: accounting for disapearance of uppermost fresh snow layer difficult due to non-constant decay rate. Unsure how to implement.

    # Get current snowheight from layer height:
    h = GRID.get_total_snowheight()

    # Check if snow or ice:
    if GRID.get_node_density(0) <= constants.snow_ice_threshold:
        if surface_temperature >= constants.zero_temperature:
            # Snow albedo decay timescale (t*) on a melting snow surface:
            t_star = constants.t_star_wet
        else:
            # Snow albedo decay timescale (t*) on a dry snow surface:
            if surface_temperature < constants.t_star_cutoff:
                t_star = (
                    constants.t_star_dry
                    + (constants.zero_temperature - constants.t_star_cutoff)
                    * constants.t_star_K
                )
            else:
                t_star = (
                    constants.t_star_dry
                    + (constants.zero_temperature - surface_temperature)
                    * constants.t_star_K
                )

        # Effect of snow albedo decay due to the temporal metamorphosis of snow (Bougamont et al. 2005 - based off Oerlemans & Knap 1998):
        # Exponential function discretised in order to account for variable surface temperature-dependant decay timescales.
        albedo_snow = (
            albedo_snow - (albedo_snow - constants.albedo_firn) / t_star * dt_days
        )

        # Reset if snowfall in current timestep
        if hours_since_snowfall == 0:
            albedo_snow = constants.albedo_fresh_snow

        # Effect of surface albedo decay due to the snow depth (Oerlemans & Knap 1998):
        alphaMod = albedo_snow + (constants.albedo_ice - albedo_snow) * np.exp(
            (-1.0 * h) / (constants.albedo_mod_snow_depth / 100.0)
        )

    else:
        # If no snow cover than set albedo to ice albedo
        alphaMod = constants.albedo_ice

    # Ensure output value is of the float data type.
    alphaMod = float(alphaMod)

    return alphaMod, albedo_snow


### idea; albedo decay like (Brock et al. 2000)? or?
### Schmidt et al 2017 >doi:10.5194/tc-2017-67, 2017 use the same albedo parameterisation from Oerlemans and Knap 1998 with a slight updated implementation of considering the surface temperature?
