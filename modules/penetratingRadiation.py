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

    # Store the total subsurface melt
    subsurface_melt = 0.0

    # Absorption of shortwave radiation
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

    # List with layer numbers to be removed
    list_of_layers_to_remove = []

    for idxNode in range(0, GRID.number_nodes - 1):

        # New temperature due to penetrating shortwave radiation
        T_rad = float(GRID.get_node_temperature(idxNode) + (E[idxNode] /\
                    (GRID.get_node_density(idxNode) * spec_heat_ice)) * \
                    (dt / GRID.get_node_height(idxNode)))

        if (T_rad-zero_temperature>0.0):

            # Temperature difference between layer and freezing temperature
            dT = T_rad - zero_temperature

            # Compute conversion factor A
            A = (spec_heat_ice*ice_density)/(water_density*lat_heat_melting)

             # Changes in volumetric contents
            dtheta_w = A * dT * GRID.get_node_ice_fraction(idxNode)
            dtheta_i = (water_density/ice_density) * -dtheta_w

            # If enough energy to remove layer
            if (dtheta_i>=GRID.get_node_ice_fraction(idxNode)):
                list_of_layers_to_remove.append(idxNode)
            # otherwise  
            else:
                GRID.set_node_liquid_water_content(idxNode, \
                    GRID.get_node_liquid_water_content(idxNode)+dtheta_w)
                GRID.set_node_ice_fraction(idxNode, \
                    GRID.get_node_ice_fraction(idxNode)+dtheta_i) 
                GRID.set_node_temperature(idxNode, zero_temperature)

            subsurface_melt += dtheta_w

    # Remove layers which have been melted
    GRID.remove_node(list_of_layers_to_remove)

    return subsurface_melt, Si
