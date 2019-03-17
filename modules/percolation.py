import numpy as np
from constants import *
from config import *
import logging

def percolation(GRID, water, t, debug_level):
    """ Percolation and refreezing of melt water through the snow- and firn pack

    Args:

        GRID    ::  GRID-Structure 
        water   ::  Melt water (m w.e.q.) at the surface
        dt      ::  Integration time

    """

    # Start logging
    logger = logging.getLogger(__name__)

    # initial runoff [m w.e.]
    Q = 0  
    

    # set liquid water of top layer (idx, LWCnew) in m
    GRID.set_node_liquid_water(0, GRID.get_node_liquid_water(0)+float(water))

    # for consistency check
    total_start = np.sum(GRID.get_liquid_water())

    # Loop over all internal grid points for percolation 
    for idxNode in range(0, GRID.number_nodes-1, 1):
        
        # Get irreducible water content [-]
        theta_e = GRID.get_node_irreducible_water_content(idxNode)
        
        # Get liquid water content [-]
        theta_w = GRID.get_node_liquid_water_content(idxNode)

        # Residual volume fraction of water (m^3 which is equal to m)
        residual = (theta_w - theta_e) * GRID.get_node_height(idxNode)

        if residual > 0:
            # than percolate to the next layer (add to the next layer)
            GRID.set_node_liquid_water(idxNode, GRID.get_node_liquid_water(idxNode)-residual)
            GRID.set_node_liquid_water(idxNode+1, GRID.get_node_liquid_water(idxNode+1)+residual)
        
    Q = GRID.get_node_liquid_water(GRID.number_nodes-1)
    GRID.set_node_liquid_water(GRID.number_nodes-1, 0.0)
    
    # for consistency check
    total_end = np.sum(GRID.get_liquid_water())

    # Check mass consistency
    if (total_start-total_end-Q) > 1e-8:
        logger.error('Percolation module is not mass consistent')

    return Q




