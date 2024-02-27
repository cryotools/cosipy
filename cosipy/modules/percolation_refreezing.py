import numpy as np
from numba import njit
from constants import water_percolation_method, z_lim,zero_temperature, spec_heat_ice, spec_heat_water, spec_heat_air, ice_density, air_density, water_density, lat_heat_melting, snow_ice_threshold

def percolation_refreezing(GRID, surface_water):

    # ==============================================
    # Distribute surface water to sub-surface nodes:
    # ==============================================

    # Import Sub-surface Grid Information:
    z = np.asarray(GRID.get_depth())
    h = np.asarray(GRID.get_height()) 

    # Reset output values:
    water_refrozen = 0
    Q = 0

    # Maximum snow fractional ice content:
    phi_ice_max = (snow_ice_threshold - air_density) / (ice_density - air_density)

    if water_percolation_method == 'bucket':

        # Add all water to the uppermost sub-surface layer:
        water = surface_water / GRID.get_node_height(0)
        GRID.set_node_liquid_water_content(0, GRID.get_node_liquid_water_content(0) + float(water))

    if water_percolation_method == 'Marchenko17':

        # Calculate the Gaussian Probability Density Function (PDF):
        PDF_normal = 2 * ((np.exp(- (z**2)/(2 * (z_lim / 3)**2))) / ((z_lim/3) * np.sqrt(2 * np.pi)))
        # Adjust in accordance with sub-surface layer heights:
        PDF_normal_height = PDF_normal * h
        # Normalise by dividing by the cumulative sum:
        Normalise = PDF_normal_height / np.sum(PDF_normal_height)
        # Update layer water content:
        water = np.asarray(GRID.get_liquid_water_content()) + ((Normalise * surface_water)/ h)
        GRID.set_liquid_water_content(water)

    # Loop over all sub-surface grid nodes:
    for Idx in range(0, GRID.number_nodes - 1, 1):

        # =======================
        # Sub-surface Refreezing:
        # =======================

        if ((GRID.get_node_temperature(Idx) - zero_temperature < 0) & (GRID.get_node_liquid_water_content(Idx) > 0)):

            # Available water for refreezing:
            available_water = GRID.get_node_liquid_water_content(Idx)

            # Volumetric/density limit on refreezing:
            d_phi_water_max_density = (phi_ice_max - GRID.get_node_ice_fraction(Idx)) * (ice_density/water_density)

            # Temperature difference between layer and freezing temperature, cold content in temperature
            dT_max = abs(GRID.get_node_temperature(Idx) - zero_temperature) # (Positive T)

            # Compute conversion factor (1/K)
            Conversion = ((spec_heat_ice * ice_density) / (water_density * lat_heat_melting))

            # Cold content limit on refreezing:  
            d_phi_water_max_coldcontent = (GRID.get_node_ice_fraction(Idx) * Conversion * dT_max) / (1 - (Conversion * dT_max * (water_density / ice_density)))
            """
            # Considers the temperature increase of unfrozen water remaining in the layer
            #d_phi_water_max_coldcontent = (dT_max * ((GRID.get_node_ice_fraction(Idx) * ice_density * spec_heat_ice) + (GRID.get_node_liquid_water_content(Idx) * water_density * spec_heat_water))) / (water_density * ((lat_heat_melting) - (dT_max * spec_heat_ice) + (dT_max * spec_heat_water)))
            """
            # Water refreeze amount:
            d_phi_water = min(available_water,d_phi_water_max_density,d_phi_water_max_coldcontent)

            # Update sub-surface node volumetric ice fraction and liquid water content:
            GRID.set_node_liquid_water_content(Idx, (GRID.get_node_liquid_water_content(Idx) - d_phi_water)) 
            d_phi_ice = d_phi_water * (water_density/ice_density)
            GRID.set_node_ice_fraction(Idx, (GRID.get_node_ice_fraction(Idx) + d_phi_ice))

            # Update sub-surface node temperature for latent heat release:
            dT = d_phi_water / (Conversion * GRID.get_node_ice_fraction(Idx))
            """
            # Including layer unfrozen water content (compatible with zero ice fraction layers)
            #dT = (d_phi_water * water_density * lat_heat_melting) / ((spec_heat_ice * ice_density * GRID.get_node_ice_fraction(Idx)) + (spec_heat_water * water_density * GRID.get_node_liquid_water_content(Idx)))
            """
            GRID.set_node_temperature(Idx, GRID.get_node_temperature(Idx) + dT)

        else:
            d_phi_water = 0
            d_phi_ice = 0

        # Record amount of refreezing:
        GRID.set_node_refreeze(Idx, d_phi_ice * GRID.get_node_height(Idx))
        water_refrozen =  water_refrozen + (d_phi_water * GRID.get_node_height(Idx))

        # ========================
        # Water Storage & Run-off:
        # ========================

        # Irreducible water content:
        phi_irreducible = GRID.get_node_irreducible_water_content(Idx)
        # Liquid water content:
        phi_water = GRID.get_node_liquid_water_content(Idx)   
        # Residual volumetric fraction of water:
        residual = np.maximum((phi_water - phi_irreducible), 0.0)

        if residual > 0:
            # Set current layer as saturated (at irreducible water content):
            GRID.set_node_liquid_water_content(Idx, phi_irreducible)
            residual = residual * GRID.get_node_height(Idx)
            GRID.set_node_liquid_water_content(Idx + 1, GRID.get_node_liquid_water_content(Idx + 1) + residual / GRID.get_node_height(Idx + 1))
        else:
            # Set current layer with unsaturated water content:
            GRID.set_node_liquid_water_content(Idx, phi_water)

    # Water in the last sub-surface node is allocated to run-off:
    Q = GRID.get_node_liquid_water_content(GRID.number_nodes - 1) * GRID.get_node_height(GRID.number_nodes - 1)
    GRID.set_node_liquid_water_content(GRID.number_nodes - 1, 0.0)

    return Q , water_refrozen