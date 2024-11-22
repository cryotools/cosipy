import numpy as np
from numba import njit


@njit
def solveHeatEquation(GRID, dt):
    """Solve the heat equation on a non-uniform grid.

    Args:
        GRID (Grid): Glacier data structure.
        dt (int): Integration time [s].
    """

    # Define index arrays
    nl = GRID.get_number_layers()  # number of layers
    nl_1 = nl - 1
    nl_2 = nl - 2
    k = np.arange(1, nl_1)  # center points
    kl = np.arange(2, nl)  # lower points
    ku = np.arange(0, nl_2)  # upper points

    # Get grid spacing
    hlayers = np.asarray(GRID.get_height())  # Get layer heights
    diff = np.divide(np.add(hlayers[0:nl_1], hlayers[1:nl]), 2.0)
    hk = diff[0:nl_2]  # between z-1 and z
    hk1 = diff[1:nl_1]  # between z and z+1

    # Get thermal diffusivity [m2 s-1]
    K = np.asarray(GRID.get_thermal_diffusivity())
    Kl = (K[1:nl_1] + K[2:nl]) / 2.0
    Ku = (K[0:nl_2] + K[1:nl_1]) / 2.0

    T = np.array(GRID.get_temperature())  # Get temperature array from grid

    stab_t = 0.0
    c_stab = 0.8
    dt_stab = c_stab * (min(min(hk**2 / Ku), min(hk1**2 / Kl)) / 2)

    while stab_t < dt:
        dt_use = min(dt_stab, dt - stab_t)
        stab_t = stab_t + dt_use

        # Update the temperatures
        T[k] = T[k] + (
            (Kl * dt_use * (T[kl] - T[k]) / hk1)
            - (Ku * dt_use * (T[k] - T[ku]) / hk)
        ) / (0.5 * (hk + hk1))

    GRID.set_temperature(T)  # Write results to GRID
