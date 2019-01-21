import math
import numpy as np
from constants import *

def penetrating_radiation(GRID, SWnet, dt):

    # Total height of first layer
    total_height = GRID.get_node_height(0)

    subsurface_melt = 0

    for idx in range(0, GRID.number_nodes - 1):

    # Check, whether snow cover exits and penetrate SW radiation
        if GRID.get_node_density(0) <= snow_ice_threshold:

            # Calc penetrating SW fraction
            Si = float(SWnet) * 0.1

            # Total height of overlying snowpack
            total_height = total_height + GRID.get_node_height(idx)

            # Exponential decay of radiation
            Tmp = float(Si * math.exp(17.1 * -total_height))

        else:
            Si = SWnet * 0.2

            total_height = total_height + GRID.get_node_height(idx)

            Tmp = float(Si * math.exp(2.5 * -total_height))

        temperature_temp = float(GRID.get_node_temperature(idx) + (Tmp / (GRID.get_node_density(idx) * spec_heat_ice))
                * (dt / GRID.get_node_height(idx)))

        if temperature_temp > zero_temperature:
            available_energy = (temperature_temp - zero_temperature) * GRID.get_node_density(idx) * spec_heat_ice \
                                 * (GRID.get_node_height(idx) / dt)
            GRID.set_node_liquid_water_content(idx, available_energy * dt / (1000 * lat_heat_melting))

            subsurface_melt += available_energy * dt / (1000 * lat_heat_melting)

        GRID.set_node_temperature(idx, np.minimum(zero_temperature, float(GRID.get_node_temperature(idx) +
                                                 (Tmp / (GRID.get_node_density(idx) *
                                                         spec_heat_ice)) * (dt / GRID.get_node_height(idx)))))

    return subsurface_melt