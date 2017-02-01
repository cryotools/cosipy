#!/usr/bin/python

from Constants import snowIceThres, roughnessFreshSnow, roughnessFirn, roughnessIce

def updateRoughness(GRID, evdiff):
    """ This method updates the roughness length (Moelg et al 2009, J.Clim.)"""

    # Check whether snow or ice
    if (GRID.get_rho_node(0) <= snowIceThres):
    
        # Roughness length
        sigma = min(roughnessFreshSnow + 0.026 * evdiff, roughnessFirn)
        
    else:
    
        # Roughness length, set to ice
        sigma = roughnessIce


    return sigma

