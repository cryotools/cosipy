import numpy as np
from numba import njit

from cosipy.constants import Constants

# only required for njitted functions
zero_temperature = Constants.zero_temperature
spec_heat_ice = Constants.spec_heat_ice
ice_density = Constants.ice_density
water_density = Constants.water_density
snow_ice_threshold = Constants.snow_ice_threshold
lat_heat_melting = Constants.lat_heat_melting


def penetrating_radiation(GRID, SWnet: float, dt: int):
    """Get the penetrating shortwave radiation.

    Implemented methods:

        - **Bintanja95**: Bintanja & Van Den Broeke (1995)

    Args:
        GRID (Grid): Glacier data structure.
        SWnet: Incoming net shortwave radiation.
        dt: Integration time [s].

    Returns:
        tuple[float,float]: Subsurface melt and penetrating shortwave
        radiation.
    """
    penetrating_allowed = ["Bintanja95"]
    if Constants.penetrating_method == "Bintanja95":
        subsurface_melt, Si = method_Bintanja(GRID, SWnet, dt)
    else:
        msg = f'Penetrating method = "{Constants.penetrating_method}" is not allowed, must be one of {". ".join(penetrating_allowed)}'
        raise ValueError(msg)

    return subsurface_melt, Si


@njit
def method_Bintanja(GRID, SWnet: float, dt: int) -> tuple:
    """Get the penetrating shortwave radiation.

    Adapted from Bintanja & Van Den Broeke (1995).

    Args:
        GRID (Grid): Glacier data structure.
        SWnet: Incoming net shortwave radiation.
        dt: Integration time [s].

    Returns:
        tuple[float, float]: Subsurface melt and penetrating shortwave
        radiation.
    """
    subsurface_melt = 0.0  # Store the total subsurface melt

    # Absorption of shortwave radiation
    depth = np.append(
        0.0, np.array(GRID.get_depth())
    )  # numba doesn't support np.insert
    if GRID.get_node_density(0) <= snow_ice_threshold:
        Si = float(SWnet) * 0.1
        decay = np.exp(17.1 * -depth)
    else:
        Si = float(SWnet) * 0.2
        decay = np.exp(2.5 * -depth)
    E = Si * np.abs(np.diff(decay))

    list_of_layers_to_remove = []  # List with layer numbers to be removed

    # Compute conversion factor A
    A = (spec_heat_ice * ice_density) / (water_density * lat_heat_melting)

    for idxNode in range(0, GRID.number_nodes - 1):

        # New temperature due to penetrating shortwave radiation
        T_rad = float(
            GRID.get_node_temperature(idxNode)
            + (E[idxNode] / (GRID.get_node_density(idxNode) * spec_heat_ice))
            * (dt / GRID.get_node_height(idxNode))
        )

        if T_rad > zero_temperature:

            # Temperature difference between layer and freezing temperature
            dT = T_rad - zero_temperature

            """Changes in volumetric contents:
            * dtheta_w: change in water fraction
            * dtheta_i: change in ice fraction
            """
            dtheta_w = A * dT * GRID.get_node_ice_fraction(idxNode)
            dtheta_i = (water_density / ice_density) * -dtheta_w
            dh = -dtheta_i / GRID.get_node_ice_fraction(idxNode)

            if dh >= 1.0:
                list_of_layers_to_remove.append(idxNode)
            else:
                GRID.set_node_liquid_water_content(
                    idxNode,
                    GRID.get_node_liquid_water_content(idxNode) + dtheta_w,
                )
                LWC_temp = GRID.get_node_liquid_water_content(
                    idxNode
                ) * GRID.get_node_height(idxNode)
                GRID.set_node_temperature(idxNode, zero_temperature)
                GRID.set_node_height(
                    idxNode, (1 - dh) * GRID.get_node_height(idxNode)
                )
                GRID.set_node_liquid_water_content(
                    idxNode, LWC_temp / GRID.get_node_height(idxNode)
                )

            subsurface_melt = subsurface_melt + dtheta_w * GRID.get_node_height(
                idxNode
            )
        else:
            GRID.set_node_temperature(idxNode, T_rad)

    # Remove layers which have been melted
    if list_of_layers_to_remove:
        # numba jitclass can't compute fingerprint of empty list
        GRID.remove_node(list_of_layers_to_remove)

    return subsurface_melt, Si
