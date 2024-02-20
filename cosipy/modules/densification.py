import numpy as np
from constants import *
from numba import njit

def densification(GRID,SLOPE,dt):
    """Densification of the snowpack.

    Args:
        GRID    ::  GRID-Structure.
        SLOPE   ::  Slope of the surface [degrees].
        dt      ::  integration time [s].
    """

    densification_allowed = ['Boone', 'Vionnet', 'empirical', 'constant','Ligtenberg11']
    if densification_method == 'Boone':
        method_Boone(GRID,SLOPE,dt)
    elif densification_method == 'Vionnet':
        method_Vionnet(GRID,SLOPE,dt)
    if densification_method == 'Ligtenberg11':
        method_Ligtenberg(GRID,SLOPE,dt)
    elif densification_method == 'empirical':
        method_empirical(GRID,SLOPE,dt)
    elif densification_method == 'constant':
        pass
    else:
        error_msg = (
            f'Densification method = "{densification_method}"',
            "is not allowed, must be one of",
            f'{", ".join(densification_allowed)}',
        )
        raise ValueError(" ".join(error_msg))
    
@njit
def copy_layer_profiles(GRID) -> tuple:
    """Get a copy of the layer profiles.

    `np.array` returns a copy by default and is 2x faster than np.copy
    (which is not supported by numba).

    Args:
        GRID (Grid): Grid object.
    Returns:
        Profiles for height, density, temperature, liquid water content,
        and ice fraction.
    """
    heights = np.array(GRID.get_height())
    densities = np.array(GRID.get_density())
    temperatures = np.array(GRID.get_temperature())
    lwcs = np.array(GRID.get_liquid_water_content())
    ice_fractions = np.array(GRID.get_ice_fraction())

    return heights, densities, temperatures, lwcs, ice_fractions

@njit
def method_Boone(GRID,SLOPE,dt):
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
    rho0 = 150

    # Overburden snow mass
    M_s = 0.0

    # Get copy of layer heights and layer densities
    height, rho, t, lwc, icf = copy_layer_profiles(GRID)

    # Loop over all internal snow nodes
    for idxNode in range(0,GRID.get_number_snow_layers() , 1):

        if ((rho[idxNode]<snow_ice_threshold) & (height[idxNode]>minimum_snow_layer_height)):

            # Get overburden snow mass
            if (idxNode>0):
                M_s = M_s + rho[idxNode-1]*height[idxNode-1]
            elif (idxNode==0):
                M_s = M_s + rho[0]*(height[0]/2.0)

            # Viscosity
            eta = eta0 * np.exp(c4*(zero_temperature-t[idxNode])+c5*rho[idxNode])

            # Rate of change in the density
            dRho = (((M_s*9.81)/eta) + c1*np.exp(-c2*(zero_temperature-t[idxNode]) - c3*np.maximum(0.0,rho[idxNode]-rho0)))*dt
           
            # Calc changes in volumetric fractions of ice and water
            # No water in layer
            if (lwc[idxNode]==0.0):
                dtheta_i = dRho
                dtheta_w = 0.0
            # layer contains water
            else:
                dtheta_i = (dRho/2.0)
                dtheta_w = (dRho/2.0)
            
            # Set new fractions
            GRID.set_node_ice_fraction(idxNode, (1+dtheta_i) * icf[idxNode])
            GRID.set_node_liquid_water_content(idxNode, (1+dtheta_w) * lwc[idxNode])

            # Set new layer height (compaction)
            GRID.set_node_height(idxNode, (1-dRho) * height[idxNode])

            if (GRID.get_node_ice_fraction(idxNode)+GRID.get_node_liquid_water_content(idxNode)+GRID.get_node_porosity(idxNode)>1.0):
                print((1+dtheta_i)*GRID.get_node_ice_fraction(idxNode),(1+dtheta_w)*GRID.get_node_liquid_water_content(idxNode),\
                     GRID.get_node_porosity(idxNode))
                print((1+dtheta_i)*GRID.get_node_ice_fraction(idxNode)+(1+dtheta_w)*GRID.get_node_liquid_water_content(idxNode)+\
                     GRID.get_node_porosity(idxNode))
                print('Fraction > 1:',(GRID.get_node_ice_fraction(idxNode)+GRID.get_node_liquid_water_content(idxNode)+GRID.get_node_porosity(idxNode)))




def method_Vionnet(GRID,SLOPE,dt):
    """ Description: Densification through overburden stress
        after Vionnet et al. 2011
    """

    # Constants
    f2 = 1.0
    eta0 = 7.62237e6  # [N s m^-2]
    a = 0.1           # [K^-1] 
    b = 0.023         # [m^3 kg^-1]
    c = 250           # [kg m^-3]

    # Vertical Stress 
    sigma = 0.0

    # Get copy of layer heights and layer densities
    height, rho, t, lwc, icf = copy_layer_profiles(GRID)

    # Loop over all internal snow nodes
    for idxNode in range(0,GRID.get_number_snow_layers() , 1):

        if ((rho[idxNode]<snow_ice_threshold) & (height[idxNode]>minimum_snow_layer_height)):

            # Parameter f1
            f1 = 1 / (1+60.0*(lwc[idxNode]/height[idxNode]))
          
            # Snow viscosity
            eta = f1*f2*eta0*(rho[idxNode]/c)*np.exp(a*(273.14-t[idxNode])+b*rho[idxNode])

            # Vertical stress
            if (idxNode>0):
                sigma = sigma + 9.81*np.cos(SLOPE)*rho[idxNode-1]*height[idxNode-1]
            elif (idxNode==0):
                # Take only half layer height of the first layer
                sigma = sigma + 9.81*np.cos(SLOPE)*rho[0]*(height[0]/2.0)

            # Rate of change for the layer height
            dD = (-sigma/eta)*dt 

            # Rate of change for the density
            dRho = dD*rho[idxNode] 
            
            # Calc changes in volumetric fractions of ice and water
            # No water in layer
            if (lwc[idxNode]==0.0):
                dtheta_i = -dD
                dtheta_w = 0.0
            # layer contains water
            else:
                dtheta_i = -dD/2.0
                dtheta_w = -dD/2.0

            # Set new volumetric fractions
            GRID.set_node_ice_fraction(idxNode, (1+dtheta_i) * icf[idxNode])
            GRID.set_node_liquid_water_content(idxNode, (1+dtheta_w) * lwc[idxNode])

            # Set new layer height (compaction)
            GRID.set_node_height(idxNode, (1+dD)*height[idxNode])

            if (GRID.get_node_ice_fraction(idxNode)+GRID.get_node_liquid_water_content(idxNode)+GRID.get_node_porosity(idxNode)>1.0):
                print((1+dtheta_i)*GRID.get_node_ice_fraction(idxNode),(1+dtheta_w)*GRID.get_node_liquid_water_content(idxNode),\
                     GRID.get_node_porosity(idxNode))
                print((1+dtheta_i)*GRID.get_node_ice_fraction(idxNode)+(1+dtheta_w)*GRID.get_node_liquid_water_content(idxNode)+\
                     GRID.get_node_porosity(idxNode))
                print('Fraction > 1: %.5f' % (GRID.get_node_ice_fraction(idxNode)+GRID.get_node_liquid_water_content(idxNode)+GRID.get_node_porosity(idxNode)))

@njit
def method_Ligtenberg(GRID,SLOPE,dt):
    """ Description: Densification based on in situ measurements of Antarctic snow compaction
        after Arthern et al. 2010 (modified by Ligtenberg et al. 2011)
    """

    # Constants
    g = 9.81            # gravitational acceleration [m s-2]
    R = 8.314           # universal gas constant [J mol-1]
    Ec = 60e3           # creep by lattice diffusion activation energy [J mol-1]
    Eg = 42.4e3         # grain growth activation energy [J mol-1]

    # Extract variables:
    rho = np.array(GRID.get_density())
    h   = np.array(GRID.get_height())
    z   = np.array(GRID.get_depth())
    T   = np.array(GRID.get_temperature())
    theta_ice =   np.array(GRID.get_ice_fraction())
    theta_water = np.array(GRID.get_liquid_water_content())

    # Convert units:
    b = accumulation    # accumulation [mm a-1]
    dt = dt / 31536000  # timestep as a fraction of a calendar year

    # Binary mask for snow/ice determination:
    mask = np.where(rho < snow_ice_threshold,1,0)

    # Gravitational Constant:
    C = np.where(rho < 550 , 0.07 * np.maximum(1.435 - 0.151 * np.log(b) , 0.25) , 0.03 * np.maximum(2.366 - 0.293 * np.log(b) , 0.25))

    """ Note: Ligtenberg's method required average annual temperature of layers - a value difficult to calculate given layer movement and remeshing.
              An approximation is made by assuming a thermal regime ressembling a trumpet curve and using linear interpolation"""
    
    # Obtain approximate temperatures from interpolation depths (domain base and depth of zero annual amplitude) 
    Tz1 = T[np.abs(z - z_zaa).argmin()]
    Tz2 = T[-1] 
    
    # Calculate linear temperature profile model constants
    m = (Tz2 - Tz1) / (z[-1] - z_zaa)
    q = Tz1 - (m * z_zaa)
    
    # Calculate approximate annual average temperature from linear gradient interpolation
    T_AVG = z * m + q
    
    # Layer density change:
    drho = mask * dt * C * b * g * (ice_density - rho) * np.exp((-Ec / (R * T)) + (Eg / (R * T_AVG)))

    # Calculate change in volumetric ice fraction & liquid water content:
    dtheta_i = np.where(theta_water == 0, drho / ice_density , drho / 2 * ice_density)
    dtheta_w = np.where(theta_water == 0, 0 , drho / 2 * water_density)    

    # Set updated volumetric ice fraction & liquid water content:
    GRID.set_ice_fraction(dtheta_i + theta_ice)
    GRID.set_liquid_water_content(dtheta_w + theta_water)

    # Set updated layer height due to compaction:
    GRID.set_height(np.maximum(minimum_snow_layer_height, h * (rho / (rho + drho))))


def method_empirical(GRID,SLOPE,dt):
    """Simple empirical snow compaction parametrization using a constant time scale."""

    rho_max = 600.0     # maximum attainable density [kg m^-3]
    #tau = 3.6e5         # empirical compaction time scale [s]
    tau = 8.0e5         # empirical compaction time scale [s]
    
    # Get copy of layer heights and layer densities
    rho = np.array(GRID.get_density())

    # Loop over all internal snow nodes
    for idxNode in range(0,GRID.get_number_snow_layers() , 1):

        # Rate of density change
        dRho = (1/tau) * (rho_max-GRID.get_node_density(idxNode)) * dt

        if ((1-(dRho/GRID.get_node_density(idxNode)))<1):
            # Set the new ice fraction
            GRID.set_node_ice_fraction(idxNode, (rho_max + (rho[idxNode]-rho_max) * np.exp(-dt/tau))/ice_density )

            # Set height change
            GRID.set_node_height(idxNode, (1-(dRho/GRID.get_node_density(idxNode)))*GRID.get_node_height(idxNode))
