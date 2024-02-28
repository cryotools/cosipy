import numpy as np
from constants import zero_temperature, spec_heat_ice, spec_heat_water, ice_density, \
                      water_density, lat_heat_melting, snow_ice_threshold
from numba import njit

@njit
def refreezing(GRID):

    # Maximum snow fractional ice content:
    phi_ice_max = (snow_ice_threshold - air_density) / (ice_density - air_density)

    # Temperature difference between layer and freezing temperature, cold content in temperature
    dT_max = np.asarray(GRID.get_temperature()) - zero_temperature

    # Volumetric/density limit on refreezing:
    dtheta_w_max_density = (phi_ice_max - np.asarray(GRID.get_ice_fraction())) * (ice_density/water_density)

    # Changes in volumetric contents, maximum amount of water that can refreeze from cold content
    dtheta_w_max_coldcontent = (dT_max * ((np.asarray(GRID.get_ice_fraction()) * ice_density * spec_heat_ice) + (np.asarray(GRID.get_liquid_water_content()) * water_density * spec_heat_water))) / (water_density * ((lat_heat_melting) - (dT_max * spec_heat_ice) + (dT_max * spec_heat_water)))

    # Water refreeze amount:
    dtheta_w = np.min(np.stack([np.asarray(GRID.get_liquid_water_content()),dtheta_w_max_density,dtheta_w_max_coldcontent]), axis = 0) 

    # Calculate change in layer ice fraction:
    dtheta_i = (water_density/ice_density) * dtheta_w

    # Set updated ice fraction and liquid water content:
    GRID.set_liquid_water_content(np.asarray(GRID.get_liquid_water_content()) - dtheta_w)
    GRID.set_ice_fraction(np.asarray(GRID.get_ice_fraction()) + dtheta_i)

    # Layer temperature change:
    dT = (dtheta_w * water_density * lat_heat_melting) / ((spec_heat_ice * ice_density * np.asarray(GRID.get_ice_fraction())) + (spec_heat_water * water_density * np.asarray(GRID.get_liquid_water_content())))
    GRID.set_temperature(np.asarray(get_temperature()) + dT)

    # Set refreezing amounts:
    GRID.set_refreeze(dtheta_i * np.asarray(GRID.get_height()))
    water_refreezed =  np.sum(dtheta_w) * np.asarray(GRID.get_height())

    return water_refreezed


