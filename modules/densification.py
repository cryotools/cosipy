import numpy as np
from constants import *
from config import *

<<<<<<< HEAD

def densification(GRID, t, debug_level):
    """ Densification of the snowpack

    Args:
=======
# Loop over all internal grid points
    ### get copy of layer heights and layer densities
    height_layers = GRID.get_height()
    density_temp = np.copy(GRID.get_density())

    for idxNode in range(0,GRID.number_nodes-1,1):

        if idxNode == 0:
            weight = (GRID.get_node_height(idxNode)*0.5*GRID.get_node_density(idxNode))
        else:
            weight = (np.nansum(height_layers[0:idxNode-1]*density_temp[0:idxNode-1]))
>>>>>>> cosipyV1.0

        GRID    ::  GRID-Structure 
        t       ::  Integration time

    """

