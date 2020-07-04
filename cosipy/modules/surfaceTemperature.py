import numpy as np
from constants import *
from cosipy.cpkernel.io import *
from scipy.optimize import minimize
import sys


def update_surface_temperature(GRID, alpha, z0, T2, rH2, p, G, u2, RAIN, SLOPE, LWin=None, N=None):
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
        RAIN    ::  RAIN (mm)
        SLOPE   ::  Slope of the surface [degree]
        LWin    ::  Incoming longwave radiation [W m^-2]
        N       ::  Fractional cloud cover [-]

    Returns:

        Li      ::  Incoming longwave radiation [W m^-2]
        Lo      ::  Outgoing longwave radiation [W m^-2]
        H       ::  Sensible heat flux [W m^-2]
        L       ::  Latent heat flux [W m^-2]
        B       ::  Ground heat flux [W m^-2]
        Qrr     ::  Rain heat flux [W m^-2]
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
                   tol=1e-1, args=(GRID, alpha, z0, T2, rH2, p, G, u2, RAIN, SLOPE, LWin, N))

    # Set surface temperature
    GRID.set_node_temperature(0, float(res.x))
 
    (Li, Lo, H, L, B, Qrr, SWnet, rho, Lv, Cs_t, Cs_q, q0, q2) = eb_fluxes(GRID, res.x, alpha, z0, T2, rH2, p, G,
                                                               u2, RAIN, SLOPE, LWin, N,)
    
    # Consistency check
    if float(res.x)>273.16:
        logger.error('Surface temperature exceeds 273.16 K')
        logger.error(GRID.get_node_temperature(0))

    # Return fluxes
    return res.fun, res.x, Li, Lo, H, L, B, Qrr, SWnet, rho, Lv, Cs_t, Cs_q, q0, q2



def eb_fluxes(GRID, T0, alpha, z0, T2, rH2, p, G, u2, RAIN, SLOPE, LWin=None, N=None):
    ''' This functions returns the surface fluxes with Monin-Obukhov stability correction.

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
        RAIN    ::  RAIN (mm)
        SLOPE   ::  Slope of the surface [degree]
        LWin    ::  Incoming longwave radiation [W m^-2]
        N       ::  Fractional cloud cover [-]

    Returns:

        Li      ::  Incoming longwave radiation [W m^-2]
        Lo      ::  Outgoing longwave radiation [W m^-2]
        H       ::  Sensible heat flux [W m^-2]
        L       ::  Latent heat flux [W m^-2]
        B       ::  Ground heat flux [W m^-2]
        Qrr     ::  Rain heat flux [W m^-2]
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

    # turbulent Prandtl number
    Pr = 0.8

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

    stability_corretctions_allowed = ['Ri', 'MO']
    # Monin-Obukhov stability correction
    if stability_correction == 'MO':
        L = 0.0
        H0 = np.inf
        diff = np.inf
        optim = True
        iter = 0

        # Optimize Obukhov length
        while optim:
            # ustar with initial condition of L == x
            ust = ustar(u2,z,z0,L)
        
            # Sensible heat flux for neutral conditions
            Cd = np.power(0.41,2.0) / np.power(np.log(z/z0) - phi_m(z,L) - phi_m(z0,L),2.0)
            Cs_t = 0.41*np.sqrt(Cd) / (np.log(z/z0t) - phi_tq(z,L) - phi_tq(z0,L))
            Cs_q = 0.41*np.sqrt(Cd) / (np.log(z/z0q) - phi_tq(z,L) - phi_tq(z0,L))
        
            # Surface heat flux
            H = rho * spec_heat_air * Cs_t * u2 * (T2-T0) * np.cos(np.radians(SLOPE))
        
            # Latent heat flux
            LE = rho * Lv * Cs_q * u2 * (q2-q0) *  np.cos(np.radians(SLOPE))
        
            # Monin-Obukhov length
            L = MO(rho, ust, T2, H)
            
            # Heat flux differences between iterations
            diff = np.abs(H0-H)
           
            # Termination criterion
            if (diff<1e-1) | (iter>5):
                optim = False
            iter = iter+1
            
            # Store last heat flux in H0
            H0 = H
  
    # Richardson-Number stability correction
    elif stability_correction == 'Ri':
        # Bulk transfer coefficient 
        Cs_t = np.power(0.41,2.0) / ( np.log(z/z0) * np.log(z/z0t) )    # Stanton-Number
        Cs_q = np.power(0.41,2.0) / ( np.log(z/z0) * np.log(z/z0q) )    # Dalton-Number
        
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

        # Sensible heat flux
        H = rho * spec_heat_air * Cs_t * u2 * (T2-T0) * phi * np.cos(np.radians(SLOPE))
        
        # Latent heat flux
        LE = rho * Lv * Cs_q * u2 * (q2-q0) * phi * np.cos(np.radians(SLOPE))

    else:
        raise ValueError("Stability correction = \"{:s}\" is not allowed, must be one of {:s}".format(stability_correction, ", ".join(stability_corretctions_allowed)))
    
    # Outgoing longwave radiation
    Lo = -surface_emission_coeff * sigma * np.power(T0, 4.0)

    # Get thermal conductivity
    lam = GRID.get_node_thermal_conductivity(0) 
   
    # Ground heat flux
    hminus = GRID.get_node_depth(1)-GRID.get_node_depth(0)
    hplus = GRID.get_node_depth(2)-GRID.get_node_depth(1)
    B = lam * (hminus/(hplus+hminus)) * \
            ((GRID.get_node_temperature(2)-GRID.get_node_temperature(1))/hplus) + (hplus/(hplus+hminus))*((GRID.get_node_temperature(1)-T0)/hminus)

    # Rain heat flux
    QRR = water_density * spec_heat_water * (RAIN/1000/dt) * (T2 - T0)

    # Return surface fluxes
    return (float(Li), float(Lo), float(H), float(LE), float(B), float(QRR), float(SWnet), rho, Lv, Cs_t, Cs_q, q0, q2)


def phi_m(z,L):
    """ Stability function for the momentum flux.
    """
    if (L>0):
        if ((z/L)>0.0) & ((z/L)<=1.0):
            return (-5*z/L)
        elif ((z/L)>1.0):
            return (1-5) * (1+np.log(z/L)) - (z/L) 
    elif L<0:
        x = np.power((1-16*z/L),0.25)
        return 2*np.log((1+x)/2.0) + np.log((1+np.power(x,2.0))/2.0) - 2*np.arctan(x) + np.pi/2.0
    else:
        return 0.0


def phi_tq(z,L):
    """ Stability function for the heat and moisture flux.
    """
    if (L>0):
        if ((z/L)>0.0) & ((z/L)<=1.0):
            return (-5*z/L)
        elif ((z/L)>1.0):
            return (1-5) * (1+np.log(z/L)) - (z/L) 
    elif L<0:
        x = np.power((1-19.3*z/L),0.25)
        return 2*np.log((1+np.power(x,2.0))/2.0)
    else:
        return 0.0


def ustar(u2,z,z0,L):
    """ Friction velocity. 
    """
    return (0.41*u2) / (np.log(z/z0)-phi_m(z,L))


def MO(rho, ust, T2, H):
    """ Monin-Obukhov length
    """
    # Monin-Obukhov length
    if H!=0:
        return (rho*spec_heat_air*np.power(ust,3)*T2)/(0.41*9.81*H)
    else:
        return 0.0



def eb_optim(T0, GRID, alpha, z0, T2, rH2, p, G, u2, RAIN, SLOPE, LWin=None, N=None):
    ''' Optimization function to solve for surface temperature T0 '''

    # Get surface fluxes for surface temperature T0
    (Li,Lo,H,L,B,Qrr, SWnet,rho,Lv,Cs_t,Cs_q,q0,q2) = eb_fluxes(GRID, T0, alpha, z0, T2, rH2, p, G,
                                                               u2, RAIN, SLOPE, LWin, N)

    # Return the residual (is minimized by the optimization function)
    return np.abs(SWnet+Li+Lo+H+L+B+Qrr)



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
