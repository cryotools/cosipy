import math
import numpy as np
from constants import *
import sys

def default_method(GRID, SWnet, dt):
    # Total height of first layer
    total_height = 0.0

    subsurface_melt = 0.0

    ### melt push to next layer
    melt_surplus = 0.0

    ### LWC push to next layer
    LWC_surplus = 0.0

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

        list_of_layers_to_remove = []

        if temperature_temp > zero_temperature:
            available_energy = (temperature_temp - zero_temperature) * GRID.get_node_density(idx) * spec_heat_ice \
                               * (GRID.get_node_height(idx) / dt)

            ### added from layer before
            available_energy += melt_surplus

            GRID.set_node_liquid_water_content(idx, available_energy * dt / (1000 * lat_heat_melting))

            ### added from layer before
            GRID.set_node_liquid_water_content(idx, GRID.get_node_liquid_water_content(idx) + LWC_surplus)

            subsurface_melt += available_energy * dt / (1000 * lat_heat_melting)

            # Hom much energy required to melt entire layer
            melt_max = GRID.get_node_height(idx) * (GRID.get_node_density(idx) / 1000)

            if melt_max > subsurface_melt:
                # Convert melt (m w.e.) to height (m)
                height_remove = subsurface_melt / (GRID.get_node_density(idx) / 1000)
                # print('remove melt height')
                # print(GRID.get_node_height(idx))
                GRID.set_node_height(idx, GRID.get_node_height(idx) - height_remove)
                # print(GRID.get_node_height(idx),'\n')

            else:
                melt_surplus = subsurface_melt - melt_max
                LWC_surplus = GRID.get_node_liquid_water_content(idx)
                list_of_layers_to_remove.append(idx)
                # print("remove layer")

        GRID.set_node_temperature(idx, np.minimum(zero_temperature, float(GRID.get_node_temperature(idx) + \
                            (Tmp / (GRID.get_node_density(idx) * spec_heat_ice)) * (dt / GRID.get_node_height(idx)))))

    # Remove layers which have been melted
    GRID.remove_node(list_of_layers_to_remove)

    return subsurface_melt, Si

def penetrating_radiation(GRID, SWnet, dt):

    if penetrating_method == 'Bintanja95':
        subsurface_melt, Si = default_method(GRID, SWnet, dt)

    else:
        print('Penetrating radiation parameterisation ', penetrating_method, ' not available, using default')
        subsurface_melt, Si = default_method(GRID, SWnet, dt)

    return subsurface_melt, Si