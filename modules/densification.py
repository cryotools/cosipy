import numpy as np
from constants import *
import sys

def densification(GRID,SLOPE):
    """ Densification of the snowpack
    Args:
        GRID    ::  GRID-Structure
    """

    if densification_method == 'Herron80':
        method_Herron(GRID,SLOPE)

    else:
        print('ERROR: Densification parameterisation ', densification_method, ' not available, using defaul')
        method_Herron(GRID,SLOPE)

def method_Herron(GRID,SLOPE):
    # % Description: Densification through overburden pressure
    # % after Reijmer & Hock (2007) and Herron & Langway (1980)
    # % RETURNS:
    # % rho_snow   :: densitiy profile after densification    [m3/kg]
    # % h_diff     :: difference in height before and after densification [m]

    # Loop over all internal grid points
    ### get copy of layer heights and layer densities
    height_layers = GRID.get_height()
    density_temp = np.copy(GRID.get_density())

    for idxNode in range(0, GRID.number_nodes - 1, 1):

        if idxNode == 0:
            weight = (GRID.get_node_height(idxNode) * 0.5 * GRID.get_node_density(idxNode))
        else:
            weight = (np.nansum(height_layers[0:idxNode] * density_temp[0:idxNode]))

        weight *= np.cos(np.radians(SLOPE))

        if GRID.get_node_density(idxNode) < snow_firn_threshold:
            density_temp[idxNode] = GRID.get_node_density(idxNode) + K0 * np.exp(
                -E0 / (R * GRID.get_node_temperature(idxNode))) \
                                    * weight * ((ice_density - GRID.get_node_density(idxNode)) / ice_density)

        elif (GRID.get_node_density(idxNode) > snow_firn_threshold) and (GRID.get_node_density(idxNode) < ice_density):
            density_temp[idxNode] = GRID.get_node_density(idxNode) + K1 * np.exp(
                -E1 / (R * GRID.get_node_temperature(idxNode))) \
                                    * weight * ((ice_density - GRID.get_node_density(idxNode)) / ice_density)
        else:
            density_temp[idxNode] = ice_density

    GRID.set_height((GRID.get_density() / density_temp) * GRID.get_height())
    GRID.set_density(density_temp)