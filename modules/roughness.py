from constants import snow_ice_threshold, roughness_fresh_snow, roughness_firn, roughness_ice, aging_factor_roughness

def updateRoughness(GRID, evdiff):
    """ This method updates the roughness length (Moelg et al 2009, J.Clim.)"""

    # Check whether snow or ice
    if (GRID.get_node_density(0) <= snow_ice_threshold):

        # Roughness length linear increase from 0.24 (fresh snow) to 4 (firn) in 60 days (1440 hours); (4-0.24)/1440 = 0.0026
        sigma = min(roughness_fresh_snow + aging_factor_roughness * evdiff, roughness_firn)
        
    else:
    
        # Roughness length, set to ice
        sigma = roughness_ice


    return (sigma/1000)

