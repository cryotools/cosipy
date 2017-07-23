#!/usr/bin/python

import math
from constants import *


def penetratingRadiation(GRID, SWnet, dt):
    """ This methods calculates the surface energy balance """

    # Total height of first layer
    totalHeight = GRID.get_hlayer_node(0)

    # Check, whether snow cover exits and penetrate SW radiation
    if (GRID.get_rho_node(0) <= snowIceThres):

        # Calc penetrating SW fraction
        Si = float(SWnet) * 0.1

        # Loop over all internal layers
        for idx in range(1,GRID.nnodes-1):

            # Total height of overlying snowpack
            totalHeight = totalHeight + GRID.get_hlayer_node(idx)

            # Exponential decay of radiation
            Tmp = float(Si * math.exp(17.1 * -totalHeight))

            # Update temperature
            GRID.set_T_node(idx, float(GRID.get_T_node(idx) + (Tmp / (GRID.get_rho_node(idx)*c_p)) * (dt/GRID.get_hlayer_node(idx))))
    else:
        Si = SWnet * 0.2
        for idx in range(1,GRID.nnodes-1):
            totalHeight = totalHeight + GRID.get_hlayer_node(idx)
            Tmp = float(Si * math.exp(2.5 * -totalHeight))
            GRID.set_T_node(idx, float(GRID.get_T_node(idx) + (Tmp / (GRID.get_rho_node(idx)*c_p)) * (dt/GRID.get_hlayer_node(idx))))
    
