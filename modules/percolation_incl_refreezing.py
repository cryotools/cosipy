import numpy as np
from constants import *
from config import *


def percolation(GRID, water, t):
    """ Percolation and refreezing of melt water through the snow- and firn pack

    t  ::  integration time/ length time step

    """

    # Courant-Friedrich Lewis Criteria
    curr_t = 0
    Tnew = 0

    Q = 0  # initial runoff [m w.e.]

    # Get height of snow layers
    hlayers = GRID.get_height()

    # Check stability criteria for diffusion
    dt_stab = c_stab * min(hlayers) / perolation_velocity

    ### array for refreezing per layer, filled with zeros
    nodes_freezing = np.zeros(GRID.number_nodes)

    ### potential freezing (>0) depending on densities, layer heights and temperatures
    potential_freezing_or_melting = spec_heat_ice * np.array(GRID.get_density()) * np.array(GRID.get_height()) / \
                            (lat_heat_melting*water_density) * (zero_temperature-np.array(GRID.get_temperature()))

    ### idices freezing
    idx_freezing = np.where(potential_freezing_or_melting > 0)

    # array with only the potential freezing layers
    nodes_potential_freezing=potential_freezing_or_melting[idx_freezing]

    runoff = 0
    # upwind scheme to adapt liquid water content of entire GRID
    while curr_t < t:

        # Select appropriate time step
        dt_use = min(dt_stab, t - curr_t)

        # set liquid water content of top layer (idx, LWCnew)
        GRID.set_node_liquid_water_content(0, (float(water) / t) * dt_use)

        # Get a copy of the GRID liquid water content profile
        liquid_water_content_time_step = np.copy(GRID.get_liquid_water_content())

        runoff_time_step=0
        # Loop over all internal grid points
        for idxNode in range(1, GRID.number_nodes-1, 1):
            # Percolation
            if (GRID.get_node_liquid_water_content(idxNode - 1) - GRID.get_node_liquid_water_content(idxNode)) != 0:
                ux = (GRID.get_node_liquid_water_content(idxNode - 1) - GRID.get_node_liquid_water_content(idxNode)) / \
                     np.abs((GRID.get_node_height(idxNode - 1) / 2.0) + (GRID.get_node_height(idxNode) / 2.0))

                uy = (GRID.get_node_liquid_water_content(idxNode + 1) - GRID.get_node_liquid_water_content(idxNode)) / \
                     np.abs((GRID.get_node_height(idxNode + 1) / 2.0) + (GRID.get_node_height(idxNode) / 2.0))

                # Calculate new liquid water content
                liquid_water_content_time_step[idxNode] = GRID.get_node_liquid_water_content(idxNode) + dt_use * (ux * perolation_velocity + uy * perolation_velocity)

        #     if idxNode == GRID.number_nodes-2:
        #         runoff_time_step = uy * perolation_velocity * dt_use
        #         print("stopS")
        #         if runoff_time_step > 0:
        #             print("runoff exist")

        # see if liquid water content is lower or potential freezing and use the minimum for refreezing at time step
        nodes_freezing_time_step = np.minimum(liquid_water_content_time_step[idx_freezing],nodes_potential_freezing)

        # add current freezing to nodes freezing (freezing per iteration)
        nodes_freezing[idx_freezing] += nodes_freezing_time_step

        # enlarge layer density if there is refreezing correct? problem with density frozen water? how change density?
        GRID.set_density((np.array(GRID.get_density()) * np.array(GRID.get_height()) + \
                                  liquid_water_content_time_step)/np.array(GRID.get_height()))

        # substract frozen water from liquid water content
        liquid_water_content_time_step[idx_freezing] -= nodes_freezing_time_step

        # Update GRID with new liquid water content tmp minus refrozen
        GRID.set_liquid_water_content(liquid_water_content_time_step)

        runoff += runoff_time_step

        # Add the time step to current time
        curr_t = curr_t + dt_use

        del liquid_water_content_time_step, nodes_freezing_time_step

    # update layer termpature when water is frozen temperature must be higher because after node potential freezing is lower
    # after one hour the termpature is raised; when there is no refreezing the temperatures has to be the same as before
    GRID.set_temperature(zero_temperature - (((potential_freezing_or_melting-nodes_freezing)*lat_heat_melting*water_density) / \
                             (np.array(GRID.get_density()) * np.array(GRID.get_height()) * spec_heat_ice)))

    return nodes_freezing, runoff