from numba import njit

from cosipy.constants import Constants

zero_temperature = Constants.zero_temperature
air_density = Constants.air_density
ice_density = Constants.ice_density
water_density = Constants.water_density
spec_heat_ice = Constants.spec_heat_ice
spec_heat_water = Constants.spec_heat_water
lat_heat_melting = Constants.lat_heat_melting
snow_ice_threshold = Constants.snow_ice_threshold


@njit
def check_oob(ice_fraction: float, lwc: float):
    """Check layer mass is conserved.

    Args:
        ice_fraction: Layer's volumetric ice fraction.
        lwc: Layer's liquid water content.

    Raises:
        ValueError: Ice fraction out of bounds.
        ValueError: Liquid water content out of bounds.
    """

    if not 0.0 <= ice_fraction <= 1.0:
        raise ValueError("Ice fraction OOB")
    elif not 0.0 <= lwc <= 1.0:
        raise ValueError("LWC OOB")


@njit
def refreezing(GRID) -> float:
    """Refreeze water in layers.

    This approach is adapted from Bartelt & Lehning (2002).

    .. math::

        \\Delta\\theta_{i} &= -\\Delta\\theta_{w}\\frac{
            \\rho_{w}
                }{
            \\rho_{i}
            }

        \\Delta T &= \\frac{
            c_{w} \\rho_{w} \\Delta\\theta_{w}
                }{
            c_{i} \\rho_{i} \\theta_{i} + c_{w} \\rho_{w} \\theta_{w}
            }

        \\theta_{refrozen} &= \\Delta\\theta_{w} h

    For the maximum water available for refreezing, the latent energy
    release from refreezing water equals the layer warming of the
    layer's ice content, the newly frozen water, and the remaining water
    that cannot be refrozen:

    .. math::
        Q_{refreeze} &= \\theta_{i}
            + \\Delta\\theta_{i}
            + (\\theta_{w} - \\Delta\\theta_{w})

        \\Delta\\theta_{w} L_{f} \\rho_{w} &= \\Delta T_{max}
            \\left [
                (c_{i} \\rho_{i}(\\theta_{i} + \\Delta\\theta_{i}))
                + (c_{w} \\rho_{w} (\\theta_{w} - \\Delta\\theta_{w}))
            \\right ]

        \\Delta\\theta_{w} L_{f} \\rho_{w} &= \\Delta T_{max}
            \\left [
                \\left (
                    c_{i} \\rho_{i}
                    \\left (
                        \\theta_{i} + \\frac{
                            \\rho_{w}
                                }{
                            \\rho_{i}
                            }\\Delta\\theta_{w}
                    \\right )
                \\right )
                + (c_{i} \\rho_{w}(\\theta_{w} - \\Delta\\theta_{w}))
            \\right ]

    Re-arranged in terms of :math:`\\Delta\\theta_{w}`, limited by the
    maximum cold content:

    .. math::

        \\Delta\\theta_{{w}_{max}} = \\frac{
            -\\Delta T_{max}(\\rho_{i}c_{i}\\theta_{i}
            + \\rho_{w}c_{w}\\theta_{w})
                }{
            \\rho_{w}(L_{f}-\\Delta T_{max}(c_{i}-c_{w}))
            }

    .. note::

        The units for :math:`\\Delta\\theta_{w} h` cancel out to m w.e.
        as long as the density of waer is set to 1000 kg m^-3.
        Note that :math:`\\Delta\\theta_{i} h` is in m **ice**
        equivalent, but both the refreeze parameter and returned
        refrozen water are in m w.e.

    Args:
        GRID (Grid): Glacier data structure.

    Returns:
        Refrozen water, [|m w.e.|].
    """

    # Maximum snow fractional ice content:
    phi_ice_max = (snow_ice_threshold - air_density) / (
        ice_density - air_density
    )
    ice_water_density_ratio = ice_density / water_density
    ice_spec_density_product = spec_heat_ice * ice_density
    water_heat_density_product = water_density * lat_heat_melting
    refrozen_water = 0.0
    for idx in range(GRID.number_nodes):
        if (GRID.get_node_temperature(idx) <= zero_temperature) & (
            GRID.get_node_liquid_water_content(idx) > 0.0
        ):
            ice_fraction = GRID.get_node_ice_fraction(idx)
            lwc = GRID.get_node_liquid_water_content(idx)
            temperature = GRID.get_node_temperature(idx)
            dT_max = temperature - zero_temperature

            # Volumetric/density limit
            dtheta_w_max_density = max(
                (phi_ice_max - ice_fraction) * ice_water_density_ratio, 0.0
            )

            # Cold content limit, dT_max is negative
            dtheta_w_max_coldcontent = -(
                dT_max
                * (
                    (ice_fraction * ice_spec_density_product)
                    + (lwc * water_density * spec_heat_water)
                )
            ) / (
                water_density
                * (
                    lat_heat_melting
                    - dT_max * (spec_heat_ice - spec_heat_water)
                )
            )
            dtheta_w_bulk = (
                -(ice_spec_density_product * ice_fraction * dT_max)
                / water_heat_density_product
            )

            # Maximum refrozen water, dtheta_w >= 0.0
            dtheta_w = min(
                (
                    lwc,
                    dtheta_w_max_density,
                    dtheta_w_max_coldcontent,
                    dtheta_w_bulk,
                )
            )

            # Change in the layer's ice fraction
            dtheta_i = (water_density / ice_density) * dtheta_w

            # Update ice fraction and liquid water content
            check_oob(ice_fraction=ice_fraction + dtheta_i, lwc=lwc - dtheta_w)
            GRID.set_node_liquid_water_content(idx, lwc - dtheta_w)
            GRID.set_node_ice_fraction(idx, ice_fraction + dtheta_i)

            # Layer temperature change
            dT = (dtheta_w * water_heat_density_product) / (
                (ice_spec_density_product * GRID.get_node_ice_fraction(idx))
                + (
                    spec_heat_water
                    * water_density
                    * (GRID.get_node_liquid_water_content(idx))
                )
            )
            GRID.set_node_temperature(idx, temperature + dT)
            if temperature + dT < 0.0:
                raise ValueError("Temperature OOB")
        else:
            dtheta_i = 0.0
            dtheta_w = 0.0
        height = GRID.get_node_height(idx)

        # Set refreezing
        # dtheta_i * ice_density = dtheta_w * water_density
        delta_mwe = dtheta_w * height
        GRID.set_node_refreeze(idx, delta_mwe)
        refrozen_water += delta_mwe

    return refrozen_water
