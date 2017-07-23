#!/usr/bin/python

import numpy as np
from constants import *
from dyn.io import *
from scipy.optimize import minimize

def energyBalance(x, GRID, SWnet, rho, Cs, T2, u2, q2, p, Li, phi, lam):

    if x > 273.16:
        Lv = L_mv
    else:
        Lv = L_ms

    # Saturation vapour pressure at the surface
    Ew0 = 6.107 * np.exp((9.5*(x-273.16)) / (265.5 + (x-273.16)))

    # Sensible heat flux
    H = rho * c_p * Cs * u2 * (x - T2) * phi

    # Mixing ratio at surface
    q0 = (100.0 * 0.622 * (Ew0 / (p - Ew0))) / 100.0

    # Latent heat flux
    L = rho * Lv * Cs * u2 * (q0 - q2) * phi

    # Outgoing longwave radiation
    Lo = -eps * sigma * np.power(x, 4.0)

    # Ground heat flux
    B = -lam * ((2.0 * GRID.get_T_node(1) - (0.5 * ((3.0 * x) + GRID.get_T_node(2)))) /
                (GRID.get_hlayer_node(0)))


    # Return residual of energy balance
    return np.abs(SWnet+Li+Lo-H-L-B)




def updateSurfaceTemperature(GRID, alpha, z0, t):
    """ This methods updates the surface temperature and returns the surface fluxes 
    
    GRID    :: GRID class
    alpha   :: Albedo
    t       :: Current time step

    """

    # todo change input data when input file format is changed to netCDF!
    T2 = DATA['T2'][t]
    rH2 = DATA['rH2'][t]
    N = DATA['N'][t]
    p = DATA['p'][t]
    G = DATA['G'][t]
    u2 = DATA['u2'][t]

    T2 = 275.5

    # Saturation vapour pressure
    Ew = 6.107 * np.exp((9.5*(T2-273.16)) / (265.5 + (T2-273.16)))

    # Water vapour at 2 m
    Ea = (rH2 * Ew) / 100.0

    # Incoming longwave radiation CORRECT
    eps_cs = 0.23 + 0.433 * np.power(Ea/T2,1.0/8.0)
    eps_tot = eps_cs * (1 - np.power(N,2)) + 0.984 * np.power(N,2)
    Li = eps_tot * sigma * np.power(T2,4.0)

    # Relative humidity to mixing ratio
    q2 = (rH2 * 0.622 * (Ew/(p-Ew))) / 100.0

    # Air density 
    rho = (p*100.0) / (287.058 * (T2 * (1 + 0.608 * q2)))

    # Bulk Richardson number todo: Check if eq correct!
    Ri = (9.81 * (T2-zeroT)*2.0) / (T2 * np.power(u2,2))

    # Stability correction
    if (Ri > 0.01) & (Ri <= 0.2):
        phi = np.power(1-5*Ri,2)
    elif Ri > 0.2:
        phi = 0
    else:
        phi = 1


    # Total net shortwave radiation
    SWnet = G * (1-alpha)

    # Bulk transfer coefficient todo: I think z0e-4 is wrong it must be z0e-3
    Cs = np.power(0.41,2.0) / np.power(np.log(2.0/(z0/1000)),2)

    # Get mean snow density
    if (GRID.get_rho_node(0) >= 830.):
        snowRhoMean = snowIceThres
    else:
        snowRho = [idx for idx in GRID.get_rho() if idx<=830.]  
        snowRhoMean = sum(snowRho)/len(snowRho)

    # Calculate thermal conductivity [W m-1 K-1] from mean density
    lam = 0.021 + 2.5 * (snowRhoMean/1000.0)**2.0
    
    res = minimize(energyBalance, GRID.get_T_node(0), method='L-BFGS-B', bounds=((200.0, 273.16),), tol=1e-8,  \
            args=(GRID, SWnet, rho, Cs, T2, u2, q2, p, Li, phi, lam))
 
    # Set surface temperature
    GRID.set_T_node(0, float(res.x))

    if res.x > 273.16:
        Lv = L_mv
    else:
        Lv = L_ms

    # Sensible heat flux
    H = rho * c_p * Cs * u2 * (res.x-T2) * phi

    # Saturation vapour pressure at the surface
    Ew0 = 6.107 * np.exp((9.5*(res.x-273.16)) / (265.5 + (res.x-273.16)))

    # Mixing ratio at surface
    q0 = (100.0 * 0.622 * (Ew0/(p-Ew0))) / 100.0

    # Latent heat flux
    L = rho * Lv * Cs * u2 * (q0-q2) * phi

    # Outgoing longwave radiation
    Lo = -eps * sigma * np.power(res.x, 4.0)

    # Ground heat flux
    B = -lam * ((2 * GRID.get_T_node(1) - (0.5 * ((3 * res.x) + GRID.get_T_node(2)))) /
                (GRID.get_hlayer_node(0)))

    return res.fun, res.x, Li, Lo, H, L, B, SWnet


