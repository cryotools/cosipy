import numpy as np
from numpy.distutils.command.install_clib import install_clib
import logging

from constants import *
from config import *

def solveHeatEquation(GRID, t):
    """ Solves the heat equation on a non-uniform grid using

    dt  ::  integration time
    
    """
    # start module logging
    logger = logging.getLogger(__name__)

    curr_t = 0
    Tnew = 0

    # Calculate thermal conductivity [W m-1 K-1] from mean density
    #lam = 0.021 + 2.5 * (np.asarray(GRID.get_density())/1000.0)**2.0

    # Calculate thermal diffusivity [m2 s-1]
    K = GRID.get_thermal_diffusivity() #lam / (np.asarray(GRID.get_density()) * np.asarray(GRID.get_specific_heat()))
    
    # Get snow layer heights    
    hlayers = np.asarray(GRID.get_height())

    # Check stability criteria for diffusion
    dt_stab = min(c_stab * ((hlayers)**2.0) / (K))

    if dt_stab<250:
        dt_stab = 180

    if max(GRID.get_temperature())>273.2:
        logger.error('Input temperature data exceeds 273.2 K')
        logger.error(GRID.get_temperature())

    while curr_t < t:
   
        # Get a copy of the GRID temperature profile
        Ttmp = np.copy(GRID.get_temperature())

        # Loop over all internal grid points
        for idxNode in range(1,GRID.number_nodes-1,1):

            # Calculate thermal diffusivity [m2 s-1]
            K = GRID.get_node_thermal_diffusivity(idxNode) #lam / (rho * cp)

            # Grid spacing            
            hk = ((hlayers[idxNode]/2.0)+(hlayers[idxNode-1]/2.0))
            hk1 = ((hlayers[idxNode+1]/2.0)+(hlayers[idxNode]/2.0))
            
            # Lagrange coeffiecients
            ak = 2.0 / (hk*(hk+hk1))
            bk = -2.0 / (hk*hk1)
            ck = 2.0 / (hk1*(hk+hk1))

            # Select appropriate time step
            dt_use = min(dt_stab,t-curr_t)

            # Calculate new temperatures
            Tnew = Ttmp[idxNode] + (K * dt_use) * \
                (ak * Ttmp[idxNode-1] + bk * Ttmp[idxNode] + ck * Ttmp[idxNode+1]) 

            # Update GRID with new temperatures
            GRID.set_node_temperature(idxNode, Tnew)

        # Add the time step to current time
        curr_t = curr_t + dt_use
    
    if max(GRID.get_temperature())>273.2:
       logger.error('Temperature exceeds 273.2 K')
       logger.error(Ttmp)
       logger.error(GRID.get_temperature())
            
