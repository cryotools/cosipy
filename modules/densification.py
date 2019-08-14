import numpy as np
from constants import *
from config import *
import sys

def densification(GRID,SLOPE):
    """ Densification of the snowpack
    Args:
        GRID    ::  GRID-Structure
    """

    if densification_method == 'Essery2013_empirical':
        method_Essery_empirical(GRID,SLOPE)
    elif densification_method == 'Essery2013_physical':
        method_Essery_physical(GRID,SLOPE)
    elif densification_method == 'Boone':
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
                dtheta_i = new_rho/ice_density
                dtheta_w = 0.0
                #dtheta_i = (new_rho/2.0)/ice_density
                #dtheta_w = (new_rho/2.0)/water_density

            # Set new fractions
            GRID.set_node_ice_fraction(idxNode, (1+dtheta_i) * GRID.get_node_ice_fraction(idxNode))
            GRID.set_node_liquid_water_content(idxNode, (1+dtheta_w) * GRID.get_node_liquid_water_content(idxNode))

            #GRID.set_node_height(idxNode, (GRID.get_node_density(idxNode) / (new_rho+rho[idxNode])) * GRID.get_node_height(idxNode))

            if (GRID.get_node_ice_fraction(idxNode)+GRID.get_node_liquid_water_content(idxNode)+GRID.get_node_porosity(idxNode)>1.0):
                print((1+dtheta_i)*GRID.get_node_ice_fraction(idxNode),(1+dtheta_w)*GRID.get_node_liquid_water_content(idxNode),\
                     GRID.get_node_porosity(idxNode))
                print((1+dtheta_i)*GRID.get_node_ice_fraction(idxNode)+(1+dtheta_w)*GRID.get_node_liquid_water_content(idxNode)+\
                     GRID.get_node_porosity(idxNode))
                print('Fraction > 1: %.5f' % (GRID.get_node_ice_fraction(idxNode)+GRID.get_node_liquid_water_content(idxNode)+GRID.get_node_porosity(idxNode)))




def method_Essery_empirical(GRID,SLOPE):
    """ Description: Densification through overburden pressure
        after Essery et al. 2013

        RETURNS:
        rho_snow   :: densitiy profile after densification    [m3/kg]
        h_diff     :: difference in height before and after densification [m]
    """

    # Get copy of layer heights and layer densities
    density_temp = np.copy(GRID.get_density())

    # Constants
    tau = 1.065954e6   # s-1
    density_max = 400  # kg/m3

    # Loop over all internal snow nodes
    for idxNode in range(0,GRID.get_number_snow_layers() - 1 , 1):
        # No densification if density is above defined maximum density
        if (GRID.get_node_density(idxNode)<density_max):
            density_temp[idxNode]=(GRID.get_node_density(idxNode) - density_max) * np.exp(-dt/tau) + density_max

            # Calc changes in volumetric fractions of ice and water
            # No water in layer
            if (GRID.get_node_liquid_water_content(idxNode)==0.0):
                dtheta_i = (density_temp[idxNode]-GRID.get_node_density(idxNode))/ice_density
                dtheta_w = 0.0
            # layer contains water
            else:
                dtheta_i = ((density_temp[idxNode]-GRID.get_node_density(idxNode))/2.0)/ice_density
                dtheta_w = ((density_temp[idxNode]-GRID.get_node_density(idxNode))/2.0)/water_density

            # Set new fractions
            GRID.set_node_ice_fraction(idxNode, (1+dtheta_i) * GRID.get_node_ice_fraction(idxNode))
            GRID.set_node_liquid_water_content(idxNode, (1+dtheta_w) * GRID.get_node_liquid_water_content(idxNode))

    GRID.set_height((GRID.get_density() / density_temp) * GRID.get_height())




def method_Essery_physical(GRID,SLOPE):
    """ Description: Densification through overburden pressure
        after Essery et al. 2013

        RETURNS:
        rho_snow   :: densitiy profile after densification    [m3/kg]
        h_diff     :: difference in height before and after densification [m]
    """

    # Get copy of layer heights and density
    height_layers = GRID.get_height()
    density_temp = np.copy(GRID.get_density())

    # Constants
    g = 9.81        # m s^-2
    c1 = 2.8e-6     # s^-1
    c2 = 0.042      # K^-1
    c3 = 0.046
    c4 = 0.081      # K^-1
    c5 = 0.018      # m^3 kg^-1
    eta0 = 3.7e7    # kg m^-1 s^-1
    Tm = 273.15     # K
    rho0 = 150      # kg m^-3

    # Loop over internal snow nodes
    for idxNode in range(0, GRID.get_number_snow_layers() - 1, 1):

        # Get overburden pressure
        if idxNode == 0:
            weight = (GRID.get_node_height(idxNode) * 0.5 * GRID.get_node_density(idxNode))
        else:
            weight = (np.nansum(height_layers[0:idxNode] * density_temp[0:idxNode]))

        weight *= np.cos(np.radians(SLOPE))

        # Solve equation
        def rhs(rho, weight, temperature):
            y1 = rho * (weight * g / (eta0 * np.exp(c4 * (Tm - temperature) + c5 * rho)) + c1 * np.exp(
                -c2 * (Tm - temperature) - c3 * np.max([0, (rho - rho0)])))
            return y1

        k1 = rhs(GRID.get_node_density(idxNode), weight, GRID.get_node_temperature(idxNode))
        k2 = rhs(GRID.get_node_density(idxNode) + dt * k1 / 2., weight, GRID.get_node_temperature(idxNode))
        k3 = rhs(GRID.get_node_density(idxNode) + dt * k2 / 2., weight, GRID.get_node_temperature(idxNode))
        k4 = rhs(GRID.get_node_density(idxNode) + dt * k3, weight, GRID.get_node_temperature(idxNode))
        density_temp[idxNode] = GRID.get_node_density(idxNode) + (dt / 6.) * (k1 + 2 * k2 + 2 * k3 + k4)

        # Calc changes in volumetric fractions of ice and water
        # No water in layer
        if (GRID.get_node_liquid_water_content(idxNode)==0.0):
            dtheta_i = (density_temp[idxNode]-GRID.get_node_density(idxNode))/ice_density
            dtheta_w = 0.0
        # layer contains water
        else:
            dtheta_i = ((density_temp[idxNode]-GRID.get_node_density(idxNode))/2.0)/ice_density
            dtheta_w = ((density_temp[idxNode]-GRID.get_node_density(idxNode))/2.0)/water_density

        # Set new fractions
        GRID.set_node_ice_fraction(idxNode, (1+dtheta_i) * GRID.get_node_ice_fraction(idxNode))
        GRID.set_node_liquid_water_content(idxNode, (1+dtheta_w) * GRID.get_node_liquid_water_content(idxNode))

    GRID.set_height((GRID.get_density() / density_temp) * GRID.get_height())




