import numpy as np
from constants import zero_temperature, spec_heat_ice, ice_density, \
                      water_density, lat_heat_melting
from numba import njit
from cosipy.utils.options import read_opt

def refreezing(GRID, opt_dict=None):

    # Read and set options
    read_opt(opt_dict, globals())

    refreezing_method = 'default'
    refreezing_allowed = ['default']

    if refreezing_method == 'default':
        water_refreezed = refreezing_default(GRID)
    else:
        raise ValueError("Refreezing method = \"{:s}\" is not allowed, must be one of {:s}".format(refreezing_method, ", ".join(refreezing_allowed)))

    return water_refreezed

#@njit
def refreezing_default(GRID):

    # water refreezed
    water_refreezed = 0.0
    LWCref = 0.0

    # Irreducible water when refreezed
    theta_r = 0.0

    #numba expects to sum numpty types
    total_start = np.sum(np.array(GRID.get_liquid_water_content()))

    # Loop over all internal grid points for percolation
    for idxNode in range(0, GRID.number_nodes-1, 1):

        if ((GRID.get_node_temperature(idxNode)-zero_temperature<1e-3) & (GRID.get_node_liquid_water_content(idxNode)>theta_r)):

            # Temperature difference between layer and freezing temperature, cold content in temperature
            dT = GRID.get_node_temperature(idxNode) - zero_temperature

            # Compute conversion factor A (1/K)
            A = (spec_heat_ice*ice_density)/(water_density*lat_heat_melting)

            # Changes in volumetric contents, maximum amount of water that can refreeze from cold content
            dtheta_w = A * dT * GRID.get_node_ice_fraction(idxNode)

            # Check if enough water water to refreeze, if less water than potential energy from cold content, only available water is refreezed
            if ((GRID.get_node_liquid_water_content(idxNode)+dtheta_w) < theta_r):
                dtheta_w = theta_r - GRID.get_node_liquid_water_content(idxNode)

            dtheta_i = (water_density/ice_density) * -dtheta_w
            dT       = dtheta_i / A
            GRID.set_node_temperature(idxNode, GRID.get_node_temperature(idxNode)+dT)

            if ((GRID.get_node_ice_fraction(idxNode)+dtheta_i+theta_r) >= 1.0):
                GRID.set_node_liquid_water_content(idxNode, theta_r)
                GRID.set_node_ice_fraction(idxNode, 1.0)
            else:
                GRID.set_node_liquid_water_content(idxNode, GRID.get_node_liquid_water_content(idxNode)+dtheta_w)
                GRID.set_node_ice_fraction(idxNode, GRID.get_node_ice_fraction(idxNode)+dtheta_i)

        else:

            dtheta_i = 0
            dtheta_w = 0

        GRID.set_node_refreeze(idxNode, dtheta_i*GRID.get_node_height(idxNode))
        water_refreezed =  water_refreezed - dtheta_w * GRID.get_node_height(idxNode)

    total_end = np.sum(np.array(GRID.get_liquid_water_content()))

    return water_refreezed


