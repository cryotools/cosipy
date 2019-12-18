import numpy as np
from constants import *
from cosipy.cpkernel.io import *
from scipy.optimize import minimize
import sys


def update_surface_temperature(GRID, alpha, z0, T2, rH2, p, G, u2, SLOPE, LWin=None, N=None):
    """ This methods updates the surface temperature and returns the surface fluxes

    Given:

        GRID    ::  Grid structure
        T0      ::  Surface temperature [K]
        alpha   ::  Albedo [-]
        z0      ::  Roughness length [m]
        T2      ::  Air temperature [K]
        rH2     ::  Relative humidity [%]
        p       ::  Air pressure [hPa]
        G       ::  Incoming shortwave radiation [W m^-2]
        u2      ::  Wind velocity [m S^-1]
        SLOPE   ::  Slope of the surface [degree]
        LWin    ::  Incoming longwave radiation [W m^-2]
        N       ::  Fractional cloud cover [-]

    Returns:

        Li      ::  Incoming longwave radiation [W m^-2]
        Lo      ::  Outgoing longwave radiation [W m^-2]
        H       ::  Sensible heat flux [W m^-2]
        L       ::  Latent heat flux [W m^-2]
        B       ::  Ground heat flux [W m^-2]
        SWnet   ::  Shortwave radiation budget [W m^-2]
        rho     ::  Air density [kg m^-3]
        Lv      ::  Latent heat of vaporization [J kg^-1]
        Cs_t    ::  Stanton number [-]
        Cs_q    ::  Dalton number [-]
        q0      ::  Mixing ratio at the surface [kg kg^-1]
        q2      ::  Mixing ratio at measurement height [kg kg^-1]
        phi     ::  Stability correction term [-]
    """
 
    # start module logging
    logger = logging.getLogger(__name__)

    # Get surface temperture by minimizing the energy balance function (SWnet+Li+Lo+H+L=0)
    res = minimize(eb_optim, GRID.get_node_temperature(0), method='L-BFGS-B', bounds=((220.0, 273.16),),
                   tol=1e-8, args=(GRID, alpha, z0, T2, rH2, p, G, u2, SLOPE, LWin, N))

    # Set surface temperature
    GRID.set_node_temperature(0, float(res.x))
 
    (Li,Lo,H,L,B,SWnet,rho,Lv,Cs_t,Cs_q,q0,q2,phi) = eb_fluxes(GRID, res.x, alpha, z0, T2, rH2, p, G,
                                                               u2, SLOPE, LWin, N,) 

    # Consistency check
    if float(res.x)>273.16:
        logger.error('Surface temperature exceeds 273.16 K')
        logger.error(GRID.get_node_temperature(0))

    # Return fluxes
    return res.fun, res.x, Li, Lo, H, L, B, SWnet, rho, Lv, Cs_t, Cs_q, q0, q2, phi



def eb_fluxes(GRID, T0, alpha, z0, T2, rH2, p, G, u2, SLOPE, LWin=None, N=None):
    ''' This functions returns the surface fluxes 

    Given:

        GRID    ::  Grid structure
        T0      ::  Surface temperature [K]
        alpha   ::  Albedo [-]
        z0      ::  Roughness length [m]
        T2      ::  Air temperature [K]
        rH2     ::  Relative humidity [%]
        p       ::  Air pressure [hPa]
        G       ::  Incoming shortwave radiation [W m^-2]
        u2      ::  Wind velocity [m S^-1]
        SLOPE   ::  Slope of the surface [degree]
        LWin    ::  Incoming longwave radiation [W m^-2]
        N       ::  Fractional cloud cover [-]

    Returns:

        Li      ::  Incoming longwave radiation [W m^-2]
        Lo      ::  Outgoing longwave radiation [W m^-2]
        H       ::  Sensible heat flux [W m^-2]
        L       ::  Latent heat flux [W m^-2]
        B       ::  Ground heat flux [W m^-2]
        SWnet   ::  Shortwave radiation budget [W m^-2]
        rho     ::  Air density [kg m^-3]
        Lv      ::  Latent heat of vaporization [J kg^-1]
        Cs_t    ::  Stanton number [-]
        Cs_q    ::  Dalton number [-]
        q0      ::  Mixing ratio at the surface [kg kg^-1]
        q2      ::  Mixing ratio at measurement height [kg kg^-1]
        phi     ::  Stability correction term [-]
    '''

    # Saturation vapour pressure (hPa)
    if saturation_water_vapour_method == 'Sonntag90':
        Ew = method_EW_Sonntag(T2)
        Ew0 = method_EW_Sonntag(T0)
    else:
        print('Method for saturation water vapour ', saturation_water_vapour_method, ' not available, using default')
        Ew = method_EW_Sonntag(T2)
        Ew0 = method_EW_Sonntag(T0)
    
    # latent heat of vaporization
    if T0 >= zero_temperature:
        Lv = lat_heat_vaporize
    else:
        Lv = lat_heat_sublimation

    # Water vapour at height z in  m (hPa)
    Ea = (rH2 * Ew) / 100.0

    # Calc incoming longwave radiation, if not available Ea has to be in Pa (Konzelmann 1994)
    if LWin is None:
        eps_cs = 0.23 + 0.433 * np.power(100*Ea/T2,1.0/8.0)
        eps_tot = eps_cs * (1 - np.power(N,2)) + 0.984 * np.power(N,2)
        Li = eps_tot * sigma * np.power(T2,4.0)
    else:
    # otherwise use LW data from file
        Li = LWin

    # Mixing Ratio at surface and at measurement height  or calculate with other formula? 0.622*e/p = q
    q2 = (rH2 * 0.622 * (Ew / (p - Ew))) / 100.0
    q0 = (100.0 * 0.622 * (Ew0 / (p - Ew0))) / 100.0
    
    # Air density 
    rho = (p*100.0) / (287.058 * (T2 * (1 + 0.608 * q2)))

    # Total net shortwave radiation
    SWnet = G * (1-alpha)

    # Bulk transfer coefficient 
    z0t = z0/100    # Roughness length for sensible heat
    z0q = z0/10     # Roughness length for moisture
    Cs_t = np.power(0.41,2.0) / ( np.log(z/z0) * np.log(z/z0t) )    # Stanton-Number
    Cs_q = np.power(0.41,2.0) / ( np.log(z/z0) * np.log(z/z0q) )    # Dalton-Number

    # Get thermal conductivity
    lam = GRID.get_node_thermal_conductivity(0) 
   
    # Bulk Richardson number
    if (u2!=0):
        Ri = (9.81 * (T2 - T0) * 2.0) / (T2 * np.power(u2, 2))
    else:
        Ri = 0
    
    # Stability correction
    if (Ri > 0.01) & (Ri <= 0.2):
        phi = np.power(1-5*Ri,2)
    elif Ri > 0.2:
        phi = 0
    else:
        phi = 1
    
    # turbulent Prandtl number
    Pr = 0.8

    # Sensible heat flux
    H = rho * spec_heat_air * (1.0/Pr) * Cs_t * u2 * (T2-T0) * phi * np.cos(np.radians(SLOPE))

    # Latent heat flux
    L = rho * Lv * (1.0/Pr) * Cs_q * u2 * (q2-q0) * phi * np.cos(np.radians(SLOPE))

    # Outgoing longwave radiation
    Lo = -surface_emission_coeff * sigma * np.power(T0, 4.0)

    # Ground heat flux
    hminus = GRID.get_node_depth(1)-GRID.get_node_depth(0)
    hplus = GRID.get_node_depth(2)-GRID.get_node_depth(1)
    B = lam * (hminus/(hplus+hminus)) * \
            ((GRID.get_node_temperature(2)-GRID.get_node_temperature(1))/hplus) + (hplus/(hplus+hminus))*((GRID.get_node_temperature(1)-T0)/hminus)

    # Return surface fluxes
    return (float(Li), float(Lo), float(H), float(L), float(B), float(SWnet), rho, Lv, Cs_t, Cs_q, q0, q2, phi)




def eb_optim(T0, GRID, alpha, z0, T2, rH2, p, G, u2, SLOPE, LWin=None, N=None):
    ''' Optimization function to solve for surface temperature T0 '''

    # Get surface fluxes for surface temperature T0
    (Li,Lo,H,L,B,SWnet,rho,Lv,Cs_t,Cs_q,q0,q2,phi) = eb_fluxes(GRID, T0, alpha, z0, T2, rH2, p, G,
                                                               u2, SLOPE, LWin, N)

    # Return the residual (is minimized by the optimization function)
    return np.abs(SWnet+Li+Lo+H+L+B)



def method_EW_Sonntag(T):
    ''' Saturation vapor pressure 
    
    Input:
        T   ::  Temperature [K]
    '''
    if T==273.16:
        # over water
        Ew = 6.112 * np.exp((17.67*(T-273.16)) / ((T-29.66)))
    else:
        # over ice
        Ew = 6.112 * np.exp((22.46*(T-273.16)) / ((T-0.55)))

    return Ew
