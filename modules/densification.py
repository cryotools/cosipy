import numpy as np
from constants import *
from config import *
import sys

def densification(GRID,SLOPE):
    """ Densification of the snowpack
    Args:
        GRID    ::  GRID-Structure
    """

    if densification_method == 'Boone':
        method_Boone(GRID,SLOPE)
    else:
        print('ERROR: Densification parameterisation', densification_method, 'not available, using default')
        method_Essery_empirical(GRID,SLOPE)



def method_Boone(GRID,SLOPE):
    """ Description: Densification through overburden pressure
        after Essery et al. 2013

        RETURNS:
        rho_snow   :: densitiy profile after densification    [m3/kg]
        h_diff     :: difference in height before and after densification [m]
    """

    # Constants
    c1 = 2.8e-6
    c2 = 0.042
    c3 = 0.046
    c4 = 0.081
    c5 = 0.018
    eta0 = 3.7e6
    rho0 = 150

    # Overburden snow mass
    M_s = 0.0

    # Get copy of layer heights and layer densities
    rho = np.copy(GRID.get_density())
    height = np.copy(GRID.get_height())

    # Loop over all internal snow nodes
    for idxNode in range(0,GRID.get_number_snow_layers() - 1 , 1):

        if (rho[idxNode]<500.0):

            # Get overburden snow mass
            if (idxNode>0):
                M_s = M_s + rho[idxNode-1]*height[idxNode-1]

            # Viscosity
            eta = eta0 * np.exp(c4*(zero_temperature-GRID.get_node_temperature(idxNode))+c5*rho[idxNode])

            # New density
            new_rho = (((M_s*9.81)/eta) + \
                       c1*np.exp(-c2*(zero_temperature-GRID.get_node_temperature(idxNode)) - \
                       c3*np.maximum(0.0,GRID.get_node_density(idxNode)-rho0)))*dt*rho[idxNode]

            # Calc changes in volumetric fractions of ice and water
            # No water in layer
            if (GRID.get_node_liquid_water_content(idxNode)==0.0):
                dtheta_i = new_rho/ice_density
                dtheta_w = 0.0
            # layer contains water
            else:
                dtheta_i = (new_rho/2.0)/ice_density
                dtheta_w = (new_rho/2.0)/water_density

            # Set new fractions
            GRID.set_node_ice_fraction(idxNode, (1+dtheta_i) * GRID.get_node_ice_fraction(idxNode))
            GRID.set_node_liquid_water_content(idxNode, (1+dtheta_w) * GRID.get_node_liquid_water_content(idxNode))

            # Set new layer height (compaction)
            GRID.set_node_height(idxNode, (rho[idxNode]/GRID.get_node_density(idxNode)) * GRID.get_node_height(idxNode))

            if (GRID.get_node_ice_fraction(idxNode)+GRID.get_node_liquid_water_content(idxNode)+GRID.get_node_porosity(idxNode)>1.0):
                print((1+dtheta_i)*GRID.get_node_ice_fraction(idxNode),(1+dtheta_w)*GRID.get_node_liquid_water_content(idxNode),\
                     GRID.get_node_porosity(idxNode))
                print((1+dtheta_i)*GRID.get_node_ice_fraction(idxNode)+(1+dtheta_w)*GRID.get_node_liquid_water_content(idxNode)+\
                     GRID.get_node_porosity(idxNode))
                print('Fraction > 1: %.5f' % (GRID.get_node_ice_fraction(idxNode)+GRID.get_node_liquid_water_content(idxNode)+GRID.get_node_porosity(idxNode)))
