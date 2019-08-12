import math
import numpy as np
from constants import *
import sys

def penetrating_radiation(GRID, SWnet, dt):

    if penetrating_method == 'Bintanja95':
        subsurface_melt, Si = method_Bintanja(GRID, SWnet, dt)

    else:
        print('Penetrating radiation parameterisation ', penetrating_method, ' not available, using default')
        subsurface_melt, Si = method_Bintanja(GRID, SWnet, dt)

    return subsurface_melt, Si


def method_Bintanja(GRID, SWnet, dt):
    # Total height of first layer
    total_height = 0.0

    subsurface_melt = 0.0

    ### melt push to next layer
    melt_surplus = 0.0

    ### LWC push to next layer
    LWC_surplus = 0.0

    
    if GRID.get_node_density(0) <= snow_ice_threshold:
        Si = float(SWnet)*0.1
        depth = np.asarray(GRID.get_depth())
        depth = np.insert(depth,0,0)
        decay = np.exp(17.1*-depth)
        E = Si*np.abs(np.diff(decay))
    else:
        Si = float(SWnet)*0.2
        depth = np.asarray(GRID.get_depth())
        depth = np.insert(depth,0,0)
        decay = np.exp(2.5*-depth)
        E = Si*np.abs(np.diff(decay))

    list_of_layers_to_remove = []

    for idx in range(0, GRID.number_nodes - 1):

        temperature_temp = float(GRID.get_node_temperature(idx) + (E[idx] / (GRID.get_node_density(idx) * spec_heat_ice))
                                 * (dt / GRID.get_node_height(idx)))

        if temperature_temp > zero_temperature:
            available_energy = (temperature_temp - zero_temperature) * GRID.get_node_density(idx) * spec_heat_ice \
                               * (GRID.get_node_height(idx) / dt)

            ### added from layer before
            #available_energy


            if (GRID.get_node_density(idx)<snow_ice_threshold):
                GRID.set_node_liquid_water(idx, available_energy * dt / (1000 * lat_heat_melting))

                ### added from layer before
                GRID.set_node_liquid_water(idx, GRID.get_node_liquid_water(idx) + LWC_surplus)

            subsurface_melt += available_energy * dt / (1000 * lat_heat_melting)
            melt = available_energy * dt / (1000 * lat_heat_melting) + melt_surplus

            # Hom much energy required to melt entire layer
            melt_max = GRID.get_node_height(idx) * (GRID.get_node_density(idx) / 1000)

            if melt_max > melt:
                # Convert melt (m w.e.) to height (m)
                height_remove = melt / (GRID.get_node_density(idx) / 1000)
                GRID.set_node_height(idx, GRID.get_node_height(idx) - height_remove)
            else:
                melt_surplus = melt - melt_max
                LWC_surplus = GRID.get_node_liquid_water(idx)
                list_of_layers_to_remove.append(idx)
        
        GRID.set_node_temperature(idx, np.minimum(zero_temperature, float(GRID.get_node_temperature(idx) + \
                            (E[idx] / (GRID.get_node_density(idx) * spec_heat_ice)) * (dt / GRID.get_node_height(idx)))))

    # Remove layers which have been melted
    GRID.remove_node(list_of_layers_to_remove)

    return subsurface_melt, Si
