import numpy as np
from constants import *
from config import *
import logging

def percolation(GRID, water, t):
    """ Percolation and refreezing of melt water through the snow- and firn pack

    Args:

        GRID    ::  GRID-Structure 
        water   ::  Melt water (m w.e.q.) at the surface
        dt      ::  Integration time

    """

    # Start logging
    logger = logging.getLogger(__name__)

    # convert m to mm = kg/m2, not needed because of change to fraction
    # water = water * 1000

    # convert kg/m2 to kg/m3
    water = water / GRID.get_node_height(0)


    # kg/m3 to fraction
    # water = water / 1000

    # initial runoff [m w.e.]
    Q = 0  
    
    # set liquid water of top layer (idx, LWCnew) in m
    GRID.set_node_liquid_water_content(0, GRID.get_node_liquid_water_content(0)+float(water))

    # for consistency check
    total_start = np.sum(GRID.get_liquid_water_content())
    
    # Loop over all internal grid points for percolation 
    for idxNode in range(0, GRID.number_nodes-1, 1):

        # Get irreducible water content [-]
        theta_e = GRID.get_node_irreducible_water_content(idxNode)
        
        # Get liquid water content [-]
        theta_w = GRID.get_node_liquid_water_content(idxNode)
    
        # Residual volume fraction of water (m^3 which is equal to m)
        residual = np.maximum((theta_w - theta_e), 0.0)

        if residual > 0:
            # than percolate to the next layer (add to the next layer)
            GRID.set_node_liquid_water_content(idxNode, theta_e)

            ### old
            #GRID.set_node_liquid_water_content(idxNode+1, GRID.get_node_liquid_water_content(idxNode+1)+residual)

            ### new: if water is pushed to next layer, because of fractions the layer heights have to be considered
            residual = residual * GRID.get_node_height(idxNode)
            GRID.set_node_liquid_water_content(idxNode + 1, GRID.get_node_liquid_water_content(idxNode + 1) + residual / GRID.get_node_height(idxNode+1))
        else: 
            GRID.set_node_liquid_water_content(idxNode, theta_w)

    # Runoff is equal to the LWC in the last node and has to be converted from kg/m3 to kg/m2
    # convert from fraction to kg/m3 (*1000) and from mm to m (/1000) not needed
    Q = GRID.get_node_liquid_water_content(GRID.number_nodes-1) * GRID.get_node_height(GRID.number_nodes-1)
    GRID.set_node_liquid_water_content(GRID.number_nodes-1, 0.0)

    # for consistency check
    total_end = np.sum(GRID.get_liquid_water_content())

    # Check mass consistency
#    if (total_start-total_end-Q) > 1e-8:
#        logger.error('Percolation module is not mass consistent')

    return Q
