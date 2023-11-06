import numpy as np
from numba import njit


@njit
def percolation(GRID, water: float, dt: float) -> float:
    """Percolation and refreezing of melt water through the snow- and firn pack

    Args:
        GRID: GRID object.
        water: Melt water at the surface, [m w.e.q.].
        dt: Integration time.

    Returns:
        float: Percolated meltwater.
    """

    # convert m to mm = kg/m2, not needed because of change to fraction
    # water = water * 1000

    # convert kg/m2 to kg/m3
    water = water / GRID.get_node_height(0)

    # kg/m3 to fraction
    # water = water / 1000

    # initial runoff [m w.e.]
    Q = 0.0

    # set liquid water of top layer (idx, LWCnew) in m
    GRID.set_node_liquid_water_content(
        0, GRID.get_node_liquid_water_content(0) + float(water)
    )

    # for consistency check
    # numba expect numpy type in np.sum()
    total_start = np.nansum(np.array(GRID.get_liquid_water_content()))

    # Loop over all internal grid points for percolation
    for idxNode in range(0, GRID.number_nodes - 1, 1):
        # Get irreducible water content [-]
        theta_e = GRID.get_node_irreducible_water_content(idxNode)

        # Get initial liquid water content [-]
        theta_w = GRID.get_node_liquid_water_content(idxNode)

        # Residual volume fraction of water (m^3 which is equal to m)
        residual = np.maximum((theta_w - theta_e), 0.0)

        if residual > 0:
            # then percolate to the next layer (add to the next layer)
            GRID.set_node_liquid_water_content(idxNode, theta_e)

            ### old
            # GRID.set_node_liquid_water_content(idxNode+1, GRID.get_node_liquid_water_content(idxNode+1)+residual)

            ### new: if water is pushed to next layer, because of fractions the layer heights have to be considered
            residual = residual * GRID.get_node_height(idxNode)
            GRID.set_node_liquid_water_content(
                idxNode + 1,
                GRID.get_node_liquid_water_content(idxNode + 1)
                + residual / GRID.get_node_height(idxNode + 1),
            )
        else:
            GRID.set_node_liquid_water_content(idxNode, theta_w)

    # Runoff is equal to the LWC in the last node and has to be converted from kg/m3 to kg/m2
    # convert from fraction to kg/m3 (*1000) and from mm to m (/1000) not needed
    Q = GRID.get_node_liquid_water_content(
        GRID.number_nodes - 1
    ) * GRID.get_node_height(GRID.number_nodes - 1)
    GRID.set_node_liquid_water_content(GRID.number_nodes - 1, 0.0)

    check_lwc_conservation(GRID, total_start, dt)  # for consistency check
    return Q

@njit
def check_lwc_conservation(GRID, start_lwc: float, dt: float):
    """Check total liquid water content is conserved.
    
    Args:
        GRID: GRID object.
        start_lwc: Initial total liquid water content.
        dt: Integration time [s].
    """
    end_lwc = np.nansum(np.array(GRID.get_liquid_water_content()))
    if not np.isclose(start_lwc, end_lwc):
        if GRID.new_snow_timestamp == 0.0:  # can't index xarrays directly with njit
            snow_time = GRID.old_snow_timestamp
        else:
            snow_time = GRID.new_snow_timestamp
        timestep = snow_time / dt
        delta = start_lwc-end_lwc
        warn_sanity = "\nWARNING: When percolating, the initial LWC is not equal to final LWC"
        # numba doesn't support warnings, and we don't want to raise an error
        print(f"{warn_sanity} at timestep {int(timestep)}. dLWC:")
        print(delta)
