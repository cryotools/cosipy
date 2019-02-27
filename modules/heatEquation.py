import numpy as np
from numpy.distutils.command.install_clib import install_clib

from constants import *
from config import *

def solveHeatEquation(GRID, t):
    """ Solves the heat equation on a non-uniform grid using

    dt  ::  integration time
    
    """
  
    curr_t = 0
    Tnew = 0

    # Get mean snow density
    if (GRID.get_node_density(0) > snow_ice_threshold):
        snowRhoMean = ice_density
    else:
        snowRho = [idx for idx in GRID.get_density() if idx <= snow_ice_threshold]
        snowRhoMean = sum(snowRho)/len(snowRho)

    # Calculate thermal conductivity [W m-1 K-1] from mean density
    lam = 0.021 + 2.5 * (min(GRID.get_density())/1000.0)**2.0

    # Calculate thermal diffusivity [m2 s-1]
    K = lam / (min(GRID.get_density()) * min(GRID.get_specific_heat()))

    # Get snow layer heights    
    hlayers = GRID.get_height()

    # Check stability criteria for diffusion
    dt_stab = c_stab * (min(hlayers)**2.0) / (K)
    
    while curr_t < t:
   
        # Get a copy of the GRID temperature profile
        Ttmp = np.copy(GRID.get_temperature())

        # Loop over all internal grid points
        for idxNode in range(1,GRID.number_nodes-1,1):

            # Get specific heat of layer (air+water+ice)
            cp = GRID.get_node_specific_heat(idxNode)
    
            # Calculate thermal conductivity [W m-1 K-1] from mean density
            lam = 0.021 + 2.5 * (GRID.get_node_density(idxNode)/1000.0)**2.0
    
            # Calculate thermal diffusivity [m2 s-1]
            K = lam / (GRID.get_node_density(idxNode) * cp)

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
    
