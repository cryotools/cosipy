from constants import snow_ice_threshold, roughness_fresh_snow, roughness_firn, roughness_ice

def updateRoughness(GRID, evdiff):
    """ This method updates the roughness length (Moelg et al 2009, J.Clim.)"""

    # Check whether snow or ice
    if (GRID.get_node_density(0) <= snow_ice_threshold):
    
        # Roughness length
        sigma = min(roughness_fresh_snow + 0.026 * evdiff, roughness_firn)
        
    else:
    
        # Roughness length, set to ice
        sigma = roughness_ice


    return sigma

