import numpy as np
from constants import *
from config import *


def percolation(GRID, water, t):
    """ Percolation and refreezing of melt water through the snow- and firn pack

    t  ::  integration time/ length time step

    """

    # Courant-Friedrich Lewis Criteria
    curr_t = 0
    Tnew = 0

    Q = 0  # initial runoff [m w.e.]

    # Get height of snow layers
    hlayers = GRID.get_height()

    # Check stability criteria for diffusion
    dt_stab = c_stab * min(hlayers) / perolation_velocity

    # JTimer.start('while')

    # upwind scheme to adapt liquid water content of entire GRID
    while curr_t < t:

        # Select appropriate time step
        dt_use = min(dt_stab, t - curr_t)

        # set liquid water content of top layer (idx, LWCnew)
        GRID.set_node_liquid_water_content(0, (float(water) / t) * dt_use)

        # Get a copy of the GRID liquid water content profile
        LWCtmp = np.copy(GRID.get_liquid_water_content())

        # Loop over all internal grid points
        for idxNode in range(1, GRID.number_nodes-1, 1):
            # Percolation
            if (GRID.get_node_liquid_water_content(idxNode - 1) - GRID.get_node_liquid_water_content(idxNode)) != 0:
                ux = (GRID.get_node_liquid_water_content(idxNode - 1) - GRID.get_node_liquid_water_content(idxNode)) / \
                     np.abs((GRID.get_node_height(idxNode - 1) / 2.0) + (GRID.get_node_height(idxNode) / 2.0))

                uy = (GRID.get_node_liquid_water_content(idxNode + 1) - GRID.get_node_liquid_water_content(idxNode)) / \
                     np.abs((GRID.get_node_height(idxNode + 1) / 2.0) + (GRID.get_node_height(idxNode) / 2.0))

                # Calculate new liquid water content
                LWCtmp[idxNode] = GRID.get_node_liquid_water_content(idxNode) + dt_use * (ux * perolation_velocity + uy * perolation_velocity)

            if idxNode == GRID.number_nodes-2:
                runoff = uy * perolation_velocity * dt_use


        # Refreezing
        for idxNode in range(1, GRID.number_nodes, 1):

            # Cold Content
            cc = -spec_heat_ice * water_density * GRID.get_node_height(idxNode) * (GRID.get_node_density(idxNode) / 1000.0) \
                 * (GRID.get_node_temperature(idxNode) - 273.16)

            energy_water = lat_heat_melting * GRID.get_node_liquid_water_content(idxNode)

            # print("CC ", cc, "energy water: ", energy_water, "LWC: ", GRID.get_LWC_node(idxNode))
            # print(GRID.get_T_node(idxNode))


        # Update GRID with new liquid water content

        # print("Runoff", runoff)
        # print(GRID.get_LWC())
        GRID.set_liquid_water_content(LWCtmp)

        # Add the time step to current time
        curr_t = curr_t + dt_use


    '''
    percolation, saturated/unsaturated, retention, refreezing, runoff
    '''

    # for idxNode in range(0,GRID.number_nodes-1,1):
    #
    #     # absolute irreducible water content
    #     LWCmin = GRID.get_LWC_node(idxNode)*LWCfrac
    #
    #     if GRID.get_LWC_node(idxNode) > LWCmin:
    #         percofreeze = True
    #         print idxNode, 'refreeze', GRID.get_T_node(idxNode)
    #     else:
    #         percofreeze = False
    #
    #     while percofreeze:
    #
    #         percofreeze = False
    #
    #         # how much water the layer can hold?
    #
    #         # how much water must/can be shifted to next layer?
    #
    #         # if T<zeroT, how much water refreezes until T>zeroT
    #
    #         # todo How to accouont for rain and its added water, melt and energy?
    #         # percolation and refreezing module in COSIMA matlab

'''
refreezing wird an die Methode aus GRID.removeMeltEnergy() angelehnt
refreezing setzt energie frei!
refreezing in layers < 273.16 K aber neue Energie beachten!

remove certain amount of refreezing meltwater (subFreeze [m w.e.]) from the LWC[idxnode] ([m w.e.])
and add node with density p_ice [kg m^-3] below the reduced node (which then is impermeable)
'''

