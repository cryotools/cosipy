from ast import Or
import numpy as np
from constants import basal_heat_flux, heat_equation_lower_boundary
from numba import njit

def solveHeatEquation(GRID, dt):

    heat_equation_lower_boundaries = ['basal_heat_flux','prescribed_temp']
    if (heat_equation_lower_boundary == 'basal_heat_flux') or (heat_equation_lower_boundary == 'prescribed_temp'):
        HeatEquation(GRID, dt)
    else:
        raise ValueError("heat_equation_lower_boundary = \"{:s}\" is not allowed, must be one of {:s}".format(heat_equation_lower_boundary, ", ".join(heat_equation_lower_boundaries)))

@njit
def HeatEquation(GRID, dt):
    """ Solves the heat equation on a non-uniform grid

    Constants:     
                dt    ::  Integration time in a model time-step (hour) [s]
                bhf   ::  Basal heat flux [W m-2]
    Inputs:
                h(z)  ::  Layer height [m]
                K(z)  ::  Layer thermal diffusivity [m2 s-1]
                k(z)  ::  Layer thermal conductivity [W m-1 K-1]
                T(z)  ::  Layer temperature [K]    
    Outputs:
                T(z)  ::  Layer temperature (updated) [K]       
    """
    
    # Get sub-surface layer properties:
    h = np.asarray(GRID.get_height())
    K = np.asarray(GRID.get_thermal_diffusivity())
    T = np.asarray(GRID.get_temperature())
    k = np.asarray(GRID.get_thermal_conductivity())
    Tnew = T.copy()
    
    # Determine integration steps required in the solver to ensure numerical stability:
    stab_t = 0.0
    c_stab = 0.8
    dt_stab  = c_stab * min(((h[1:]+h[:-1])/2)**2/(K[1:]+K[:-1]))
    
    # Numerically solve the Fourier heat equation using a finite central difference scheme in matrix form:
    while stab_t < dt:
        dt_use = np.minimum(dt_stab, dt-stab_t)
        stab_t = stab_t + dt_use

        # Update the temperatures of the intermediate sub-surface nodes:
        Tnew[1:-1] = T[1:-1] + dt_use * (((0.5 * (K[2:]  + K[1:-1]))  * (T[2:]   - T[1:-1]) / (0.5 * (h[2:]  + h[1:-1])) - \
                                        (0.5 * (K[:-2] + K[1:-1]))  * (T[1:-1] - T[:-2])  / (0.5 * (h[:-2] + h[1:-1]))) / \
                                        (0.25 * h[:-2]  + 0.5 * h[1:-1] + 0.25 * h[2:]))
        
        if heat_equation_lower_boundary == 'basal_heat_flux':

            # Update the temperature of the base sub-surface node using the basal heat flux:
            Tnew[-1]   = T[-1]   + dt_use * (((basal_heat_flux * K[-1] / k[-1]) - \
                                            ((0.5 * (K[-2] + K[-1])) * (T[-1] - T[-2])  / (0.5 * (h[-2] + h[-1])))) / \
                                            (0.25 * h[-2] + 0.75 * h[-1]))
        
        T = Tnew.copy()
        
    # Write results to GRID
    GRID.set_temperature(T)


