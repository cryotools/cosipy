#!/usr/bin/env python

from jackedCodeTimerPY import JackedTiming
import numpy as np
from Constants import *

JTimer = JackedTiming()


def percolation(GRID, water, t):
    """ Percolation and refreezing of melt water through the snow- and firn pack

    t  ::  integration time/ length time step

    """

    # Courant-Friedrich Lewis Criteria
    c_stab = 0.1
    curr_t = 0
    Tnew = 0

    Q = 0  # initial runoff [m w.e.]

    # Get height of snow layers
    hlayers = GRID.get_hlayer()

    # Check stability criteria for diffusion
    dt_stab = c_stab * min(hlayers) / Vp

    # JTimer.start('while')

    # upwind scheme to adapt liquid water content of entire GRID
    while curr_t < t:

        # Get a copy of the GRID liquid water content profile
        LWCtmp = np.copy(GRID.get_LWC())

        # Select appropriate time step
        dt_use = min(dt_stab, t - curr_t)

        # set liquid water content of top layer (idx, LWCnew)
        GRID.set_LWC_node(0, (float(water) / t) * dt_use)

        # Loop over all internal grid points
        for idxNode in range(1, GRID.nnodes - 1, 1):
            # Grid spacing
            hk = ((hlayers[idxNode] / 2.0) + (hlayers[idxNode - 1] / 2.0))
            hk1 = ((hlayers[idxNode + 1] / 2.0) + (hlayers[idxNode] / 2.0))

            # Lagrange coefficients
            ak = hk1 / (hk * (hk + hk1))
            bk = (hk1 - hk) / (hk * hk1)
            ck = hk / (hk1 * (hk + hk1))

            # Calculate new liquid water content
            LWCnew = LWCtmp[idxNode] - (Vp * dt_use) * \
                                       (ak * LWCtmp[idxNode - 1] + bk
                                        * LWCtmp[idxNode] + ck
                                        * LWCtmp[idxNode + 1])
            # todo RuntimeWarning: overflow encountered in double_scalars
            # todo RuntimeWarning: invalid value encountered in double_scalars

            # Update GRID with new liquid water content
            GRID.set_LWC_node(idxNode, LWCnew)


        # Add the time step to current time
        curr_t = curr_t + dt_use


    '''
    percolation, saturated/unsaturated, retention, refreezing, runoff
    '''

    # for idxNode in range(0,GRID.nnodes-1,1):
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

