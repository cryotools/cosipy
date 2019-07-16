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
    else:
        print('ERROR: Densification parameterisation', densification_method, 'not available, using default')
        method_Essery_empirical(GRID,SLOPE)


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
         density_temp[idxNode]=(GRID.get_node_density(idxNode) - density_max) * np.exp(-dt*tau) + density_max

    # Update grid
    GRID.set_ice_fraction(GRID.get_height() / ((GRID.get_density() / density_temp) * GRID.get_height()) * GRID.get_ice_fraction())
    GRID.set_liquid_water(GRID.get_height() / ((GRID.get_density() / density_temp) * GRID.get_height()) * GRID.get_liquid_water())
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

    # Update grid
    GRID.set_ice_fraction(GRID.get_height() / ((GRID.get_density() / density_temp) * GRID.get_height()) * GRID.get_ice_fraction())
    GRID.set_liquid_water(GRID.get_height() / ((GRID.get_density() / density_temp) * GRID.get_height()) * GRID.get_liquid_water())
    GRID.set_height((GRID.get_density() / density_temp) * GRID.get_height())




