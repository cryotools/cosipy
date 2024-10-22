import numpy as np
from numba import njit


@njit
def percolation(GRID, water: float, dt: int) -> float:
    """Percolate melt water through the snow- and firn pack.

    Bucket method (Bartelt & Lehning, 2002).

    Args:
        GRID (Grid): Glacier data structure.
        water: Melt water at the surface, [|m w.e.| q.].
        dt: Integration time [s].

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

    Q = get_runoff(GRID)

    return Q


@njit
def get_runoff(grid) -> float:
    """Get meltwater runoff for a column.

    Runoff is equal to LWC in the last node & must be converted
    from kg/m3 to kg/m2. Converting from fraction to kg/m3 (\\*1000) and
    from mm to m (/1000) is unnecessary.

    Args:
        grid (Grid): Glacier data structure.

    Returns:
        Meltwater runoff.
    """

    max_index = grid.number_nodes - 1
    runoff = grid.get_node_liquid_water_content(
        max_index
    ) * grid.get_node_height(max_index)
    grid.set_node_liquid_water_content(max_index, 0.0)

    return runoff
