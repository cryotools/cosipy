import numpy as np
from numba import njit
from cosipy.utils.options import read_opt

def solveHeatEquation(GRID, dt, opt_dict=None):

    # Read and set options
    read_opt(opt_dict, globals())

    heatEquation_method = 'default'
    heatEquation_allowed = ['default']

    if heatEquation_method == 'default':
        heatEquation_default(GRID,dt)
    else:
        raise ValueError("Heat equation = \"{:s}\" is not allowed, must be one of {:s}".format(heatEquation_method, ", ".join(heatEquation_allowed)))

@njit
def heatEquation_default(GRID, dt):
    """ Solves the heat equation on a non-uniform grid

    dt  ::  integration time
    
    """
    # number of layers
    nl = GRID.get_number_layers()

    # Define index arrays 
    k   = np.arange(1,nl-1) # center points
    kl  = np.arange(2,nl)   # lower points
    ku  = np.arange(0,nl-2) # upper points
    
    # Get thermal diffusivity [m2 s-1]
    K = np.asarray(GRID.get_thermal_diffusivity()) 
    
    # Get snow layer heights    
    hlayers = np.asarray(GRID.get_height())

    # Get grid spacing
    diff = ((hlayers[0:nl-1]/2.0)+(hlayers[1:nl]/2.0))
    hk = diff[0:nl-2]  # between z-1 and z
    hk1 = diff[1:nl-1] # between z and z+1
    
    # Get temperature array from grid|
    T = np.array(GRID.get_temperature())
    Tnew = T.copy()
    
    Kl = (K[1:nl-1]+K[2:nl])/2.0
    Ku = (K[0:nl-2]+K[1:nl-1])/2.0
    
    stab_t = 0.0
    c_stab = 0.8
    dt_stab  = c_stab * (min([min(diff[0:nl-2]**2/(2*Ku)),min(diff[1:nl-1]**2/(2*Kl))]))
    
    while stab_t < dt:

        dt_use = np.minimum(dt_stab, dt-stab_t)
        stab_t = stab_t + dt_use

        # Update the temperatures
        Tnew[k] += ((Kl*dt_use*(T[kl]-T[k])/(hk1)) - (Ku*dt_use*(T[k]-T[ku])/(hk))) / (0.5*(hk+hk1))
        T = Tnew.copy()
        
    # Write results to GRID
    GRID.set_temperature(T)
