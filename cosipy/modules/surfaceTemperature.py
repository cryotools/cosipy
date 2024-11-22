import numpy as np
from numba import njit
from scipy.optimize import minimize, newton

from cosipy.config import Config
from cosipy.constants import Constants
from cosipy.modules.secant import secant

zlt1 = Constants.zlt1
zlt2 = Constants.zlt2
saturation_water_vapour_method = Constants.saturation_water_vapour_method
zero_temperature = Constants.zero_temperature
lat_heat_vaporize = Constants.lat_heat_vaporize
lat_heat_sublimation = Constants.lat_heat_sublimation
sigma = Constants.sigma
stability_correction = Constants.stability_correction
spec_heat_air = Constants.spec_heat_air
spec_heat_water = Constants.spec_heat_water
sfc_temperature_method = Constants.sfc_temperature_method
surface_emission_coeff = Constants.surface_emission_coeff
water_density = Constants.water_density
WRF_X_CSPY = Config.WRF_X_CSPY


def update_surface_temperature(GRID, dt, z, z0, T2, rH2, p, SWnet, u2, RAIN, SLOPE, LWin=None, N=None):
    """Solve the surface temperature and get the surface fluxes.

    Implemented minimisation methods:

        - **Newton**: Recommended, and fastest.
        - **Secant**: Potentially faster for coupled WRF simulations.
        - **L-BFGS-B**: Slower, different results since 02/2024.
        - **SLSQP**: Slower, different results since 02/2024.

    Args:
        GRID (Grid): Glacier data structure.
        dt (float): Integration time [s] - can vary in WRF_X_CSPY.
        z (float): Measurement height [m] - varies in WRF_X_CSPY.
        z0 (float): Roughness length [m].
        T2 (float): Air temperature [K].
        rH2 (float): Relative humidity [%].
        p (float): Air pressure [hPa].
        SWnet (float): Incoming shortwave radiation [|W m^-2|].
        u2 (float): Wind velocity [|m s^-1|].
        RAIN (float): Rain [mm].
        SLOPE (float): Slope of the surface [degree].
        LWin (float): Incoming longwave radiation [|W m^-2|].
        N (float): Fractional cloud cover [-].

    Returns:
        tuple:
        :res.fun: Minimisation function.
        :res.x: Surface temperature [K].
        :Li: Incoming longwave radiation [|W m^-2|].
        :Lo: Outgoing longwave radiation [|W m^-2|].
        :H: Sensible heat flux [|W m^-2|].
        :L: Latent heat flux [|W m^-2|].
        :B: Ground heat flux [|W m^-2|].
        :Qrr: Rain heat flux [|W m^-2|].
        :rho: Air density [|kg m^-3|].
        :Lv: Latent heat of vaporization [|J kg^-1|].
        :MOL: Monin-Obukhov length [m].
        :Cs_t: Stanton number [-].
        :Cs_q: Dalton number [-].
        :q0: Mixing ratio at the surface [|kg kg^-1|].
        :q2: Mixing ratio at measurement height [|kg kg^-1|].

    Raises:
        NotImplementedError: Invalid method for minimizing the residual.
    """
    
    # Interpolate subsurface temperatures to selected subsurface depths for GHF computation
    B_Ts = interp_subT(GRID)
    
    # Lower bound for surface temperature
    lower_bnd_ts = 220.
    upper_bnd_ts = 330.
    initial_guess = min(GRID.get_node_temperature(0), 270)

    if sfc_temperature_method in ['L-BFGS-B', 'SLSQP']:
        # Get surface temperature by minimizing the energy balance function (SWnet+Li+Lo+H+L=0)
        res = minimize(eb_optim, initial_guess, method=sfc_temperature_method,
                       bounds=((lower_bnd_ts, upper_bnd_ts),),tol=1e-2,
                       args=(GRID, dt, z, z0, T2, rH2, p, SWnet, u2, RAIN, SLOPE, B_Ts, LWin, N))

    elif sfc_temperature_method == 'Newton':
        try:
            res = newton(eb_optim, np.array([initial_guess]), tol=1e-2, maxiter=50,
                        args=(GRID, dt, z, z0, T2, rH2, p, SWnet, u2, RAIN, SLOPE, B_Ts, LWin, N))
            check_residual_clip(residual=res, lower_bound=lower_bnd_ts)

        except (RuntimeError,ValueError):
             # Workaround for non-convergence and unboundedness
             res = minimize(eb_optim, initial_guess, method='SLSQP',
                       bounds=((lower_bnd_ts, upper_bnd_ts),),tol=1e-2,
                       args=(GRID, dt, z, z0, T2, rH2, p, SWnet, u2, RAIN, SLOPE, B_Ts, LWin, N))
    elif sfc_temperature_method == 'Secant':
        try:
            res = call_secant_jitted(
                GRID, dt, z, z0, T2, rH2, p, SWnet, u2, RAIN, SLOPE, B_Ts, lower_bnd_ts, LWin, N, 
            )
            check_residual_clip(residual=res, lower_bound=lower_bnd_ts)

        except (RuntimeError, ValueError):
            # Workaround for non-convergence and unboundedness
            res = minimize(eb_optim, initial_guess, method='SLSQP',
                           bounds=((lower_bnd_ts, zero_temperature),), tol=1e-2,
                           args=(GRID, dt, z, z0, T2, rH2, p, SWnet, u2, RAIN, SLOPE, B_Ts, LWin, N))
    else:
        raise NotImplementedError('Invalid method for minimizing the residual.')

    # Set surface temperature
    if isinstance(res, np.ndarray):
        surface_temperature = min(np.array([zero_temperature]), res)
        minimisation_function = None
    else:
        surface_temperature = min(np.array([zero_temperature]), res.x)
        minimisation_function = res.fun
    GRID.set_node_temperature(0, surface_temperature)
 
    (Li, Lo, H, L, B, Qrr, rho, Lv, MOL, Cs_t, Cs_q, q0, q2) = eb_fluxes(GRID, surface_temperature, dt,  z, z0, T2, rH2, p, u2, RAIN, SLOPE, B_Ts, LWin, N)
     
    # Consistency check
    if (surface_temperature > zero_temperature) or (surface_temperature < lower_bnd_ts):
        print('Surface temperature is outside bounds:', GRID.get_node_temperature(0))

    # Return fluxes
    return minimisation_function, surface_temperature, Li, Lo, H, L, B, Qrr, rho, Lv, MOL, Cs_t, Cs_q, q0, q2


@njit
def check_residual_clip(residual, lower_bound: float) -> bool:
    """Raise error if the surface temperature is out of bounds.
    
    Args:
        residual (float or np.ndarray): Residual from minimisation.
        lower_bound: Minimum allowable surface temperature.

    Returns:
        True if surface temperature solution is within bounds.

    Raises:
        ValueError: Surface temperature solution is out of bounds.
    """
    if residual < lower_bound:
        raise ValueError("TS solution is out of bounds.")
    return True


@njit
def call_secant_jitted(GRID, dt, z, z0, T2, rH2, p, SWnet, u2, RAIN, SLOPE, B_Ts, lower_bnd_ts, LWin=None, N=None, initial_guess=None):
    """Jitted call to secant.py.

    Returns:
        np.ndarray: Updated surface temperature.

    Raises:
        ValueError: Surface temperature solution is out of bounds.
    """
    if initial_guess is None:
        initial_guess = np.array([min(GRID.get_node_temperature(0), 270)])

    res = secant(eb_optim, initial_guess, tol=1e-2, maxiter=50,
        args=(GRID, dt, z, z0, T2, rH2, p, SWnet, u2, RAIN, SLOPE, B_Ts, LWin, N))
    if res.item() < lower_bnd_ts:
        raise ValueError("TS Solution is out of bounds")
    return res


@njit
def get_subsurface_temperature(GRID, cumulative_depth: np.ndarray, zlt: float):
    """Get subsurface temperature.

    Args:
        GRID (Grid): Glacier data structure.
        cumulative_depth: Cumulative glacier layer heights [m].
        zlt: Interpolation depth [m].

    Returns:
        float: Subsurface temperature at the interpolation depth.
    """

    # Find indexes of two depths for temperature interpolation
    idx1_depth = np.abs(cumulative_depth - zlt).argmin()
    depth = cumulative_depth.flat[idx1_depth]

    if depth > zlt:
        idx2_depth = idx1_depth - 1
    else:
        idx2_depth = idx1_depth + 1

    temperature_idx1 = GRID.get_node_temperature(idx1_depth)
    t_z = temperature_idx1 + (
        (temperature_idx1 - GRID.get_node_temperature(idx2_depth))
        / (cumulative_depth[idx1_depth] - cumulative_depth[idx2_depth])
    ) * (zlt - cumulative_depth[idx1_depth])

    return t_z


@njit
def interp_subT(GRID) -> np.ndarray:
    """Interpolate subsurface temperature to depths used for ground heat flux.

    Returns:
        Interpolated subsurface temperatures at requested depths.
    """
    
    # Cumulative layer depths
    layer_heights_cum = np.cumsum(np.array(GRID.get_height()))

    t_z1 = get_subsurface_temperature(GRID, layer_heights_cum, zlt1)
    t_z2 = get_subsurface_temperature(GRID, layer_heights_cum, zlt2)

    return np.array([t_z1, t_z2])


@njit
def get_saturation_vapor_pressure(T_0: float, T_2: float) -> tuple:
    """Get saturation vapour pressure.

    Args:
        T_0: Surface temperature [K].
        T_2: 2m air temperature [K].

    Returns:
        tuple[float, float]: Surface and 2m saturation vapour pressure.
    """

    if saturation_water_vapour_method == "Sonntag90":
        Ew = method_EW_Sonntag(T_2)
        Ew0 = method_EW_Sonntag(T_0)
    else:
        msg = (
            f"Method for saturation water vapour",
            f"{saturation_water_vapour_method}",
            "not available, using default",
        )
        print(" ".join(msg))
        Ew = method_EW_Sonntag(T_2)
        Ew0 = method_EW_Sonntag(T_0)

    return Ew, Ew0


@njit
def eb_fluxes(GRID, T0, dt, z, z0, T2, rH2, p, u2, RAIN, SLOPE, B_Ts, LWin=None, N=None):
    """Get the surface fluxes and apply the Monin-Obukhov stability correction.

    Args:
        GRID (Grid): Glacier data structure.
        T0 (float): Surface temperature [K].
        dt (float): Integration time [s].
        z (float): Measurement height [m].
        z0 (float): Roughness length [m].
        T2 (float): Air temperature [K].
        rH2 (float): Relative humidity [%].
        p (float): Air pressure [hPa].
        u2 (float): Wind velocity [|m s^-1|].
        RAIN (float): Rain [mm].
        SLOPE (float): Slope of the surface [degree].
        B_Ts (np.ndarray): Subsurface temperatures at interpolation depths [K].
        LWin (float): Incoming longwave radiation [|W m^-2|].
        N (float): Fractional cloud cover [-].

    Returns:
        tuple:
        :Li: Incoming longwave radiation [|W m^-2|].
        :Lo: Outgoing longwave radiation [|W m^-2|].
        :H: Sensible heat flux [|W m^-2|].
        :LE: Latent heat flux [|W m^-2|].
        :B: Ground heat flux [|W m^-2|].
        :QRR: Rain heat flux [|W m^-2|].
        :rho: Air density [|kg m^-3|].
        :Lv: Latent heat of vaporization [|J kg^-1|].
        :L: Monin-Obukhov length [m].
        :Cs_t: Stanton number [-].
        :Cs_q: Dalton number [-].
        :q0: Mixing ratio at the surface [|kg kg^-1|].
        :q2: Mixing ratio at measurement height [|kg kg^-1|].
    """

    # Saturation vapour pressure (hPa)
    Ew, Ew0 = get_saturation_vapor_pressure(T_0=T0, T_2=T2)
    
    # latent heat of vaporization
    if T0 >= zero_temperature:
        Lv = lat_heat_vaporize
    else:
        Lv = lat_heat_sublimation

    # Water vapour at height z in  m (hPa)
    Ea = (rH2 * Ew) / 100.0

    # Calc incoming longwave radiation, if not available Ea has to be in Pa (Konzelmann 1994)
    # numba has no implementation for power(none, int)
    if (LWin is None) and (N is not None):
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

    # Bulk transfer coefficient 
    z0t = z0/100    # Roughness length for sensible heat
    z0q = z0/10     # Roughness length for moisture
    L = None
 
    # Avoid recalculating
    slope_radians = np.radians(SLOPE)
    cos_slope_radians = np.cos(slope_radians)

    # Monin-Obukhov stability correction
    if stability_correction == 'MO':
        L = 0.0
        H0 = T0*0. + np.inf  # numba: consistent typing of H0
        diff = np.inf
        optim = True
        niter = 0

        # Optimize Obukhov length
        while optim:
            # ustar with initial condition of L == x
            ust = ustar(u2,z,z0,L)
        
            # Sensible heat flux for neutral conditions
            delta_phi_tq = phi_tq(z, L) - phi_tq(z0, L)
            Cd = np.power(0.41,2.0) / np.power(np.log(z/z0) - phi_m(z,L) - phi_m(z0,L),2.0)
            Cs_t = 0.41*np.sqrt(Cd) / (np.log(z/z0t) - delta_phi_tq)
            Cs_q = 0.41*np.sqrt(Cd) / (np.log(z/z0q) - delta_phi_tq)

            # Surface heat flux
            H = get_sensible_heat_flux(rho, Cs_t, u2, (T2 - T0), cos_slope_radians)

            # Latent heat flux
            LE = get_latent_heat_flux(rho, Lv, Cs_q, u2, (q2 - q0), cos_slope_radians)
        
            # Monin-Obukhov length
            L = MO(rho, ust, T2, H)

            # Heat flux differences between iterations
            diff = np.abs(H0-H)
           
            # Termination criterion
            if (diff<1e-1) | (niter>5):
                optim = False
            niter = niter+1
            
            # Store last heat flux in H0
            H0 = H
  
    # Richardson-Number stability correction
    elif stability_correction == 'Ri':
        # Bulk transfer coefficient 
        Cs_t = np.power(0.41,2.0) / ( np.log(z/z0) * np.log(z/z0t) )  # Stanton-Number
        Cs_q = np.power(0.41,2.0) / ( np.log(z/z0) * np.log(z/z0q) )  # Dalton-Number
        
        # Bulk Richardson number
        Ri = 0
        if u2 != 0:
            Ri = ( (9.81 * (T2 - T0) * z) / (T2 * np.power(u2, 2)) ).item() #numba can't compare literal & array below

        # Stability correction
        phi = 1
        if 0.01 < Ri <= 0.2:
            phi = np.power(1-5*Ri,2)
        elif Ri > 0.2:
            phi = 0.0

        # Sensible heat flux
        H = phi * get_sensible_heat_flux(rho, Cs_t, u2, (T2 - T0), cos_slope_radians)
        
        # Latent heat flux
        LE = phi * get_latent_heat_flux(rho, Lv, Cs_q, u2, (q2 - q0), cos_slope_radians)

    else:
        msg = f"Stability correction {stability_correction} is not supported."
        raise ValueError(msg)

    # Outgoing longwave radiation
    Lo = -surface_emission_coeff * sigma * np.power(T0, 4.0)

    # Get thermal conductivity
    lam = GRID.get_node_thermal_conductivity(0) 
   
    # Ground heat flux
    hminus = zlt1
    hplus = zlt2 - hminus  # avoid namespace collision with zlt1
    Tz1, Tz2 = B_Ts
    B = lam * ((hminus/(hplus+hminus)) * ((Tz2-Tz1)/hplus) + (hplus/(hplus+hminus)) * ((Tz1-T0)/hminus))

    # Rain heat flux
    QRR = water_density * spec_heat_water * (RAIN/1000/dt) * (T2 - T0)

    # Return surface fluxes
    # Numba: No implementation of function Function(<class 'float'>) found for signature: >>> float(array(float64, 1d, C))
    # fmt: off
    return (Li.item(), Lo.item(), H.item(), LE.item(), B.item(), QRR.item(), rho, Lv, L, Cs_t, Cs_q, q0, q2)
    # fmt: on


@njit
def get_sensible_heat_flux(
    rho: float, Cs_t: float, u2: float, dT: float, cos_slope: float
) -> float:
    """Get sensible heat flux.

    Args:
        rho: Air density, [|kg m^-3|].
        Cs_t: Stanton number, [-].
        u2: Wind velocity, [|m s^-1|].
        dT: Difference in temperature between the surface and
            measurement height, [K].
        cos_slope: Cosine of slope angle, [|rad|].

    Returns:
        Sensible heat flux, [|W m^-2|].
    """
    return rho * spec_heat_air * Cs_t * u2 * dT * cos_slope


@njit
def get_latent_heat_flux(
    rho: float, Lv: float, Cs_q: float, u2: float, dq: float, cos_slope: float
) -> float:
    """Get latent heat flux.

    Args:
        rho: Air density, [|kg m^-3|].
        Lv: Latent heat of vapourisation, [|J kg^-1|].
        Cs_q: Dalton number, [-].
        u2: Wind velocity, [|m s^-1|].
        dq: Difference in mixing ratio between the surface and
            measurement height.
        cos_slope: Cosine of slope angle, [|rad|].

    Returns:
        Latent heat flux, [|W m^-2|].
    """
    return rho * Lv * Cs_q * u2 * dq * cos_slope


@njit
def phi_m_stable(z: float, L: float) -> float:
    """Get integrated stability function for momentum, stable conditions.

    Args:
        z: Height, [m].
        L: Obukhov length, [m].

    Returns:
        Integrated stability function for momentum under stable
        conditions.
    """
    zeta = z / L
    if (zeta > 0.0) & (zeta <= 1.0):  # weak stability
        return -5 * zeta
    elif zeta > 1.0:  # strong stability
        if WRF_X_CSPY:
            return -5.0  # limit stability parameter (z/L <= 1.) following Noah-MP LSM approach
        else:
            return (1 - 5) * (1 + np.log(zeta)) - zeta
    else:
        return 0.0


@njit
def phi_m(z: float, L: float) -> float:
    """Get the integrated stability function for momentum.

    Args:
        z: Height, [m].
        L: Obukhov length, [m].

    Returns:
        Integrated stability function for momentum.
    """
    if L > 0:
        return phi_m_stable(z, L)
    elif L < 0:
        x = np.power((1 - 16 * z / L), 0.25)
        return (
            2 * np.log((1 + x) / 2.0)
            + np.log((1 + np.power(x, 2.0)) / 2.0)
            - 2 * np.arctan(x)
            + np.pi / 2.0
        )
    else:
        return 0.0


@njit
def phi_tq(z: float, L: float) -> float:
    """Stability function for the heat and moisture flux."""
    if L > 0:
        return phi_m_stable(z, L)
    elif L < 0:
        x = np.power((1 - 19.3 * z / L), 0.25)
        return 2 * np.log((1 + np.power(x, 2.0)) / 2.0)
    else:
        return 0.0


@njit
def ustar(u2: float, z: float, z0: float, L: float) -> float:
    """Get the friction velocity."""
    return (0.41 * u2) / (np.log(z / z0) - phi_m(z, L))


@njit
def MO(rho, ust, T2, H) -> float:
    """Get the Monin-Obukhov length."""
    if H!=0:
        return (
            (rho * spec_heat_air * np.power(ust, 3) * T2) / (0.41 * 9.81 * H)
        ).item()  # numba: expects a float
    else:
        return 0.0


# fmt: off
@njit
def eb_optim(
    T0: float, GRID, dt: int, z: float, z0: float, T2: float, rH2: float,
    p: float, SWnet: float, u2: float, RAIN: float, SLOPE: float, B_Ts: np.ndarray,
    LWin: float = None, N: float = None,
) -> float:
    """Optimization function to solve for surface temperature T0.

    Returns:
        Minimised residual.
    """

    # Get surface fluxes for surface temperature T0
    (Li, Lo, H, L, B, Qrr, _, _, _, _, _, _, _) = eb_fluxes(
        GRID, T0, dt, z, z0, T2, rH2, p, u2, RAIN, SLOPE, B_Ts, LWin, N
    )

    # Return the residual (minimized by the optimization function)
    if sfc_temperature_method in ['Newton', 'Secant']:
        return SWnet + Li + Lo + H + L + B + Qrr
    else:
        return np.abs(SWnet + Li + Lo + H + L + B + Qrr)
# fmt: on


@njit
def method_EW_Sonntag(T: float) -> float:
    """Get the saturation vapor pressure.

    Args:
        T: Temperature [K].

    Returns:
        Saturation vapor pressure.
    """
    if T >= 273.16:
        # over water
        Ew = 6.112 * np.exp((17.67 * (T - 273.16)) / (T - 29.66))
    else:
        # over ice
        Ew = 6.112 * np.exp((22.46 * (T - 273.16)) / (T - 0.55))
    return Ew
