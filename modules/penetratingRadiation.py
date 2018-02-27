import math
import numpy as np
from constants import *


def penetrating_radiation(GRID, SWnet, dt):
    """ This methods calculates the surface energy balance """

    # Total height of first layer
    total_height = GRID.get_node_height(0)

    # Check, whether snow cover exits and penetrate SW radiation
    if GRID.get_node_density(0) <= snow_ice_threshold:

        # Calc penetrating SW fraction
        Si = float(SWnet) * 0.1

        # Loop over all internal layers
        for idx in range(1,GRID.number_nodes-1):

            # Total height of overlying snowpack
            total_height = total_height + GRID.get_node_height(idx)

            # Exponential decay of radiation
            Tmp = float(Si * math.exp(17.1 * -total_height))

            # Update temperature
            GRID.set_node_temperature(idx, np.minimum(zero_temperature ,float(GRID.get_node_temperature(idx) +
                                                 (Tmp / (GRID.get_node_density(idx) *
                                                  spec_heat_air)) * (dt / GRID.get_node_height(idx)))))
    else:
        Si = SWnet * 0.2
        for idx in range(1, GRID.number_nodes-1):
            total_height = total_height + GRID.get_node_height(idx)
            Tmp = float(Si * math.exp(2.5 * -total_height))
            GRID.set_node_temperature(idx, np.minimum(zero_temperature, float(GRID.get_node_temperature(idx) +
                                                 (Tmp / (GRID.get_node_density(idx) *
                                                         spec_heat_air)) * (dt / GRID.get_node_height(idx)))))

