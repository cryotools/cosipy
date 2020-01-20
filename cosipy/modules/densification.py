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
    elif densification_method == 'Vionnet':
        method_Vionnet(GRID,SLOPE)
    else:
        print('ERROR: Densification parameterisation', densification_method, 'not available, using default')
        method_Boone(GRID,SLOPE)



def method_Boone(GRID,SLOPE):
    """ Description: Densification through overburden pressure
        after Essery et al. 2013
    """

    # Constants
    c1 = 2.8e-6
    c2 = 0.042
    c3 = 0.046
    c4 = 0.081
    c5 = 0.018
    eta0 = 3.7e7
    rho0 = 250

    # Overburden snow mass
    M_s = 0.0

    # Get copy of layer heights and layer densities
    rho = np.copy(GRID.get_density())
    height = np.copy(GRID.get_height())

    # Loop over all internal snow nodes
    for idxNode in range(0,GRID.get_number_snow_layers() , 1):

        if (rho[idxNode]<900.0):

            # Get overburden snow mass
            if (idxNode>0):
                M_s = M_s + rho[idxNode-1]*height[idxNode-1]
            elif (idxNode==0):
                M_s = M_s + rho[0]*(height[0]/2.0)

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



def method_Vionnet(GRID,SLOPE):
    """ Description: Densification through overburden stress
        after Vionnet et al. 2011
    """

    # Constants
    f2 = 4.0
    eta0 = 7.62237e6  # [N s m^-2]
    a = 0.1           # [K^-1] 
    b = 0.023         # [m^3 kg^-1]
    c = 250           # [kg m^-3]

    # Vertical Stress 
    sigma = 0.0

    # Get copy of layer heights and layer densities
    rho = np.copy(GRID.get_density())
    height = np.copy(GRID.get_height())

    # Loop over all internal snow nodes
    for idxNode in range(0,GRID.get_number_snow_layers() , 1):

        if (rho[idxNode]<900.0):

            # Parameter f1
            f1 = 1 / (1+60.0*GRID.get_node_liquid_water_content(idxNode))
            
            # Snow viscosity
            eta = f1*f2*eta0*(GRID.get_node_density(idxNode)/c)*np.exp(a*(273.14-GRID.get_node_temperature(idxNode))+b*GRID.get_node_density(idxNode))

            # Vertical stress
            if (idxNode>0):
                sigma = sigma + 9.81*np.cos(SLOPE)*rho[idxNode-1]*height[idxNode-1]
            elif (idxNode==0):
                # Take only half layer height of the first layer
                sigma = sigma + 9.81*np.cos(SLOPE)*rho[0]*(height[0]/2.0)

            # New Height
            new_height = GRID.get_node_height(idxNode)+GRID.get_node_height(idxNode)*(-sigma/eta)*dt

            # New density
            new_rho = (height[idxNode]/new_height)*rho[idxNode] 

            # Calc changes in volumetric fractions of ice and water
            # No water in layer
            if (GRID.get_node_liquid_water_content(idxNode)==0.0):
                dtheta_i = new_rho/ice_density
                dtheta_w = 0.0
            # layer contains water
            else:
                dtheta_i = (new_rho/2.0)/ice_density
                dtheta_w = (new_rho/2.0)/water_density

            # Set new volumetric fractions
            GRID.set_node_ice_fraction(idxNode, (1+dtheta_i) * GRID.get_node_ice_fraction(idxNode))
            GRID.set_node_liquid_water_content(idxNode, (1+dtheta_w) * GRID.get_node_liquid_water_content(idxNode))

            # Set new layer height (compaction)
            GRID.set_node_height(idxNode, new_height)

            if (GRID.get_node_ice_fraction(idxNode)+GRID.get_node_liquid_water_content(idxNode)+GRID.get_node_porosity(idxNode)>1.0):
                print((1+dtheta_i)*GRID.get_node_ice_fraction(idxNode),(1+dtheta_w)*GRID.get_node_liquid_water_content(idxNode),\
                     GRID.get_node_porosity(idxNode))
                print((1+dtheta_i)*GRID.get_node_ice_fraction(idxNode)+(1+dtheta_w)*GRID.get_node_liquid_water_content(idxNode)+\
                     GRID.get_node_porosity(idxNode))
                print('Fraction > 1: %.5f' % (GRID.get_node_ice_fraction(idxNode)+GRID.get_node_liquid_water_content(idxNode)+GRID.get_node_porosity(idxNode)))
