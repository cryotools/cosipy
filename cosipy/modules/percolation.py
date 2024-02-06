import numpy as np
from numba import njit


@njit
def percolation(GRID, water: float, dt: int) -> float:
    """Percolate melt water through the snow- and firn pack.

    Bucket method (Bartelt & Lehning, 2002).

    Args:
        GRID (Grid): Glacier data mesh.
        water: Melt water at the surface, [m w.e.q.].
        dt: Integration time.

    Returns:
        Percolated meltwater.
    """

    # convert m to mm = kg/m2, not needed because of change to fraction
    # water = water * 1000

    # convert kg/m2 to kg/m3
    water = water / GRID.get_node_height(0)
    # kg/m3 to fraction
    # water = water / 1000

    # set liquid water of top layer (idx, LWCnew) in m
    GRID.set_node_liquid_water_content(
        0, GRID.get_node_liquid_water_content(0) + float(water)
    )

    # Loop over all internal grid points for percolation
    for idxNode in range(0, GRID.number_nodes - 1, 1):
        theta_e = GRID.get_node_irreducible_water_content(idxNode)
        theta_w = GRID.get_node_liquid_water_content(idxNode)

        # Residual volume fraction of water (m^3 which is equal to m)
        residual = np.maximum((theta_w - theta_e), 0.0)

        if residual > 0.0:
            GRID.set_node_liquid_water_content(idxNode, theta_e)

            """
            old:
            GRID.set_node_liquid_water_content(
                idxNode + 1,
                GRID.get_node_liquid_water_content(idxNode + 1) + residual,
            )

            new:
            If water is pushed to next layer, the layer heights have to
            be considered because of fractions.
            """
            residual = residual * GRID.get_node_height(idxNode)
            GRID.set_node_liquid_water_content(
                idxNode + 1,
                GRID.get_node_liquid_water_content(idxNode + 1)
                + residual / GRID.get_node_height(idxNode + 1),
            )

    """Runoff is equal to LWC in the last node & must be converted
    from kg/m3 to kg/m2. Converting from fraction to kg/m3 (*1000) and
    from mm to m (/1000) is unnecessary."""
    Q = GRID.get_node_liquid_water_content(
        GRID.number_nodes - 1
    ) * GRID.get_node_height(GRID.number_nodes - 1)
    GRID.set_node_liquid_water_content(GRID.number_nodes - 1, 0.0)

    return Q


@njit
def check_lwc_conservation(GRID, start_lwc: float, runoff: float, dt: float):
    """Check total liquid water content is conserved.

    Args:
        GRID (Grid): Glacier data mesh.
        start_lwc: Initial total liquid water content.
        runoff: Meltwater runoff from the lowest node.
        dt: Integration time [s].
    """

    end_lwc = (
        np.nansum(
            np.array(GRID.get_liquid_water_content())
            * np.array(GRID.get_height())
        )
        + runoff
    )

    if not np.isclose(start_lwc, end_lwc):
        if (
            GRID.new_snow_timestamp == 0.0
        ):  # can't index xarrays directly with njit
            snow_time = GRID.old_snow_timestamp
        else:
            snow_time = GRID.new_snow_timestamp
        timestep = snow_time / dt
        delta = start_lwc - end_lwc
        warn_sanity = "\nWARNING: When percolating, the initial LWC is not equal to final LWC"
        # numba doesn't support warnings, and we don't want to raise an error
        print(f"{warn_sanity} at timestep {int(timestep)}. dLWC:")
        print(delta)
