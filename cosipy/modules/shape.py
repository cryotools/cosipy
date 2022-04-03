import numpy as np
from constants import radf, water_density, ice_density, constant_density
from scipy.optimize import minimize, newton
from numba import njit
from types import SimpleNamespace

def update_cone(GRID, MB, r_cone, h_cone, s_cone, h_old, A_cone, V_cone):
    """ This methods updates the area of the artificial ice reservoir

    Given:

        GRID    ::  Grid structure
        surfMB  ::  Surface mass balance
        r_cone  ::  Old Cone radius  [m]
        h_cone  ::  Old Cone height  [m]
        s_cone  ::  Old Cone slope  [-]
        A_cone  ::  Old Surface Area [m2]
        V_cone  ::  Old Volume [m3]

    Returns:

        r_cone  ::  New Cone radius  [m]
        h_cone  ::  New Cone height  [m]
        s_cone  ::  New Cone slope  [-]
        A_cone  ::  New Surface Area [m2]
        V_cone  ::  New Volume [m3]
    """

    delta_h = GRID.get_total_height() - h_cone
    h_cone = GRID.get_total_height()
    V_cone = 1/3 * np.pi * r_cone**2 * h_cone
    # V_cone += delta_h * A_cone
    # r_cone = np.power(3 * V_cone / (np.pi * h_cone), 1 / 2)
    # A_cone = np.pi * r_cone * np.sqrt(r_cone**2 + h_cone**2)
    # V_cone = np.pi * r_cone**2 * h_cone
    # V_cone += MB * ice_density/np.mean(GRID.get_density()) * A_cone

    if (delta_h > 0) & (r_cone >= radf):  # Maintain constant r_cone
        s_cone = h_cone / r_cone
        h_cone = (3 * V_cone / (np.pi * r_cone ** 2))
    elif (delta_h < 0) & (MB > 0):  # Due to densification
        s_cone = h_cone / r_cone
        h_cone = (3 * V_cone / (np.pi * r_cone ** 2))
    else:                               # Maintain constant slope
        r_cone = np.power(3 * V_cone / (np.pi * s_cone), 1 / 3)
        # h_cone = s_cone * r_cone

    h_old = GRID.get_total_height()
    A_cone = np.pi * r_cone * np.sqrt(r_cone**2 + h_cone**2)
    # print("Radius %.1f, Height %.01f, Volume %0.01f" %(r_cone, h_cone, V_cone))

    return r_cone, h_cone, s_cone, h_old, A_cone, V_cone
