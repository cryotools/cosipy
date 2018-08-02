import numpy as np
from constants import *
from config import *


def percolation(GRID, water, t, debug_level):
    """ Percolation and refreezing of melt water through the snow- and firn pack

    Args:

        GRID    ::  GRID-Structure 
        water   ::  Melt water (m w.e.q.) at the surface
        dt      ::  Integration time

    """

    # Courant-Friedrich Lewis Criteria
    curr_t = 0
    Tnew = 0

    Q = 0  # initial runoff [m w.e.]

    # Get height of snow layers
    hlayers = GRID.get_height()

    # Check stability criteria for diffusion
    dt_stab = c_stab * min(hlayers) / percolation_velocity

    # upwind scheme to adapt liquid water content of entire GRID
    while curr_t < t:

        # Select appropriate time step
        dt_use = min(dt_stab, t - curr_t)

        # set liquid water content of top layer (idx, LWCnew) in kg m^-2
        GRID.set_node_liquid_water_content(0, GRID.get_node_liquid_water_content(0)+((float(water)*water_density) / t) * dt_use)
       
        # Loop over all internal grid points for percolation 
        for idxNode in range(1, GRID.number_nodes-1, 1):
        
            # Percolation of water exceeding the max. retention
            if (GRID.get_node_liquid_water_content(idxNode - 1) - GRID.get_node_liquid_water_content(idxNode)) != 0:
                ux = ((1-GRID.get_node_vol_ice_content(idxNode-1))*GRID.get_node_liquid_water_content(idxNode - 1) - 
                      (1-GRID.get_node_vol_ice_content(idxNode))*GRID.get_node_liquid_water_content(idxNode)) / \
                     np.abs((GRID.get_node_height(idxNode - 1) / 2.0) + (GRID.get_node_height(idxNode) / 2.0))

                uy = ((1-GRID.get_node_vol_ice_content(idxNode+1))*GRID.get_node_liquid_water_content(idxNode + 1) - 
                      (1-GRID.get_node_vol_ice_content(idxNode))*GRID.get_node_liquid_water_content(idxNode)) / \
                     np.abs((GRID.get_node_height(idxNode + 1) / 2.0) + (GRID.get_node_height(idxNode) / 2.0))

                # Calculate new liquid water content
                if (GRID.get_node_density(idxNode)>snow_ice_threshold):
                    pvel = percolation_velocity/10.0
                else:
                    pvel = percolation_velocity
                
                GRID.set_node_liquid_water_content(idxNode, GRID.get_node_liquid_water_content(idxNode) + dt_use * (ux * pvel + uy * pvel))
                    
                # Calculate runoff in m w.e.q
                if ((GRID.get_node_density(idxNode)>=snow_ice_threshold)):
                    Q = Q + GRID.get_node_liquid_water_content(idxNode)/ice_density
                    GRID.set_node_liquid_water_content(idxNode, 0.0)
    
                # Remove water from the first layer
                if (idxNode==1):
                    GRID.set_node_liquid_water_content(0, GRID.get_node_liquid_water_content(0)- dt_use * (ux * percolation_velocity))

        
        # Add the time step to current time
        curr_t = curr_t + dt_use
   
    # Do the refreezing after percolation
    water_refreezed = refreeze(GRID)

    # Write merging steps if debug level is set >= 10
    if debug_level >= 40:
        print("Percolation ....")
        GRID.grid_info(10)
        print("End percolation .... \n")
        print("Runoff: %2.7f" % (Q))

    return Q, water_refreezed


def calc_cc(GRID, node):
    """ Calculates the cold content, the potential latent energy release by refreezing """

    # cold content of the snow at layer node (J m^-2) 
    Qcc = -spec_heat_ice * GRID.get_node_density(node) * GRID.get_node_height(node) * (GRID.get_node_temperature(node)-zero_temperature)
    GRID.set_node_cold_content(node, -Qcc)


def refreeze(GRID):

    # water refreezed
    water_refreezed = 0
    LWCref = 0

    # Loop over all internal grid points for percolation 
    for idxNode in range(0, GRID.number_nodes-1, 1):
    
        # Set cold content and potential latent heat energy
        calc_cc(GRID, idxNode)
        
        # potential latent energy if all water is refreezed (J m^-2)
        Qp = lat_heat_melting * water_density * (GRID.get_node_liquid_water_content(idxNode)/water_density)
   
        # If energy is smaller than CC, warm the snow pack
        if ((GRID.get_node_cold_content(idxNode)+Qp)<0): 

            # volumetric ice content (Coleou et Lesaffre, 1998)
            theta_ice = GRID.get_node_density(idxNode)/ice_density

            # maximum volumentric water content (Coleou et Lesaffre, 1998)
            if (theta_ice <= 0.23):
                theta_ret = 0.0264 + 0.0099*((1-theta_ice)/theta_ice) 
            elif (theta_ice > 0.23) & (theta_ice <= 0.812):
                theta_ret = 0.08 - 0.1023*(theta_ice-0.03)
            else:
                theta_ret = 0

            # Set porosity
            GRID.set_node_porosity(idxNode, 1-((GRID.get_node_density(idxNode)-theta_ret*(water_density))/ice_density))
            
            # Set volumetic ice content
            GRID.set_node_vol_ice_content(idxNode, theta_ret)
        
            # Update CC (reduce)
            GRID.set_node_cold_content(idxNode,GRID.get_node_cold_content(idxNode)+Qp)

            # Update temperature (warm the snowpack)
            GRID.set_node_temperature(idxNode, GRID.get_node_temperature(idxNode) + (Qp/(spec_heat_ice*GRID.get_node_density(idxNode)*GRID.get_node_height(idxNode))) )
            
            # Update density (increase density)
            a = GRID.get_node_height(idxNode)*(GRID.get_node_density(idxNode)/ice_density)
            b = GRID.get_node_liquid_water_content(idxNode)/water_density
            GRID.set_node_density(idxNode, (1-(b/a))*GRID.get_node_density(idxNode)+(b/a)*ice_density)

            # Set LWC to zero (remove all water)
            GRID.set_node_liquid_water_content(idxNode, 0.0) 
       
        # if refreeze energy is greater than CC, warm the snowpack to zero degrees and reduce LWC
        else:
            
            # Set temperature to zero
            GRID.set_node_temperature(idxNode, zero_temperature)
             
            # How much water is refreezed
            LWCref = np.abs(GRID.get_node_cold_content(idxNode)/(lat_heat_melting))

            # Get exceeding energy and calculate the remaing LWC
            GRID.set_node_liquid_water_content(idxNode, GRID.get_node_liquid_water_content(idxNode)-LWCref)
        
            # Update density
            a = GRID.get_node_height(idxNode)*(GRID.get_node_density(idxNode)/ice_density)
            b = LWCref/water_density
            GRID.set_node_density(idxNode, (1-(b/a))*GRID.get_node_density(idxNode)+(b/a)*ice_density)

        # Convert kg m^-2 (mm) to m w.e.q.
        water_refreezed = water_refreezed + (LWCref/ice_density)

    return water_refreezed
