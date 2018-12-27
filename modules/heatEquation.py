import numpy as np

from constants import *
from config import *

def solveHeatEquation(GRID, t):
    """ Solves the heat equation on a non-uniform grid using

    dt  ::  integration time
    
    """
  
    curr_t = 0
    Tnew = 0

    # Get mean snow density
    if (GRID.get_node_density(0) > 830.):
        #snowRhoMean = snow_ice_threshold
        snowRhoMean = soil_density
    else:
        snowRho = [idx for idx in GRID.get_density() if idx <= 830.]
        snowRhoMean = sum(snowRho)/len(snowRho)

    # Calculate specific heat  [J Kg-1 K-1] depending on mean temperature domain
    if GRID.get_total_snowheight() > 0.0:
        c_pi = 152.2 + 7.122 * np.mean(GRID.get_temperature())
    else:
        c_pi = volumetric_heat_soil / soil_density  # volumetric heat divided by bulk density of soil

    # Calculate thermal conductivity [W m-1 K-1] from mean density
    if GRID.get_total_snowheight() > 0.0:
        lam = 0.021 + 2.5 * (snowRhoMean/1000.0)**2.0
    else:
        lam = (water_content * water_thermal_conductivity ** 0.5 + mineral_content * mineral_thermal_conductivity ** \
               0.5 + air_content * air_thermal_conductivity ** 0.5) ** 2

    # Calculate thermal diffusivity [m2 s-1]
    K = lam / (snowRhoMean * c_pi)

    # Get snow layer heights    
    hlayers = GRID.get_height()

    # Check stability criteria for diffusion
    dt_stab = c_stab * (min(hlayers)**2.0) / (K)
    
    while curr_t < t:
   
        # Get a copy of the GRID temperature profile
        Ttmp = np.copy(GRID.get_temperature())

        # Loop over all internal grid points
        for idxNode in range(1,GRID.number_nodes-1,1):

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
    
    return c_pi
