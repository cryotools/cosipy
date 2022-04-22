import numpy as np
from constants import emissivity_method, snow_emission_coeff, ice_emission_coeff, surface_emission_coeff, snow_ice_threshold

def updateEmissionCoeff(GRID):
    """ This methods updates the emission coefficient """
    emissivity_allowed = ['constant', 'Balasubramanian22']
    if emissivity_method == 'constant':
        sigma = surface_emission_coeff
    elif emissivity_method == 'Balasubramanian22':
        sigma = method_Balasubramanian(GRID)
    else:
        raise ValueError("Emissivity method = \"{:s}\" is not allowed, must be one of {:s}".format(emissivity_method, ", ".join(emissivity_allowed)))
    return sigma

def method_Balasubramanian(GRID):
    if (GRID.get_node_density(0) <= snow_ice_threshold):
        sigma = snow_emission_coeff
    else:
        sigma = ice_emission_coeff
    return sigma
