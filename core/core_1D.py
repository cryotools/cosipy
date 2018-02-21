import numpy as np

from constants import *
from config import *
from confs_dicts_constants.dictionaries_1D import create_1D_output_numpy_arrays

from modules.albedo import updateAlbedo
from modules.heatEquation import solveHeatEquation
from modules.penetratingRadiation import penetrating_radiation
from modules.percolation import percolation
from modules.roughness import updateRoughness
from modules.surfaceTemperature import update_surface_temperature

import core.grid as grd

def core_1D(air_pressure, cloud_cover, initial_snow_height, relative_humidity, snowfall, solar_radiation,
                 temperature_2m, wind_speed):

    albedo_all, condensation_all, depostion_all, evaporation_all, ground_heat_flux_all, longwave_in_all, \
    longwave_out_all, latent_heat_flux_all, melt_heigt_all, number_layers_all, sensible_heat_flux_all, snowHeight_all, \
    shortwave_net_all, sublimation_all, surface_melt_all, surface_temperature_all, u2_all, sw_in_all, T2_all, rH2_all, \
    snowfall_all, pressure_all, cloud_all, sh_all, rho_all, Lv_all, Cs_all, q0_all, q2_all, qdiff_all, phi_all = \
    create_1D_output_numpy_arrays(temperature_2m)

    ''' INITIALIZATION '''
    hours_since_snowfall = 0
    # Init layers
    layer_heights = 0.1 * np.ones(number_layers)

    rho = ice_density * np.ones(number_layers)
    rho[0] = 250.
    rho[1] = 400.
    rho[2] = 550.

    # Init temperature
    temperature_surface = temperature_bottom * np.ones(number_layers)
    for i in range(len(temperature_surface)):
        gradient = ((temperature_2m[0] - temperature_bottom) / number_layers)
        temperature_surface[i] = temperature_2m[0] - gradient * i

    # Init liquid water content
    liquid_water_content = np.zeros(number_layers)

    # if merging_level == 0:
    #     print('Merging level 0')
    # else:
    #     print('Merge in action!')

    # Initialize grid, the grid class contains all relevant grid information
    GRID = grd.Grid(layer_heights, rho, temperature_surface, liquid_water_content, debug_level)
    # todo params handling?

    # Get some information on the grid setup
    GRID.info()

    # Merge grid layers, if necessary
    GRID.update_grid(merging_level)

    if initial_snow_height[0]:
        snow_height = initial_snow_height
    else:
        snow_height = 0

    # inital mass balance
    mass_balance = 0

    ' TIME LOOP '

    for t in range(time_start, time_end, 1):
        print(t)
        # Add snowfall
        snow_height = snow_height + snowfall[t]

        if snowfall[t] > 0.0:
            # TODO: Better use weq than snowheight

            # Add a new snow node on top
            GRID.add_node(float(snowfall[t]), density_fresh_snow, float(temperature_2m[t]), 0)
            GRID.merge_new_snow(merge_snow_threshold)

        if snowfall[t] < 0.005:
            hours_since_snowfall += dt / 3600.0
        else:
            hours_since_snowfall = 0

        # Calculate albedo and roughness length changes if first layer is snow
        # Update albedo values
        alpha = updateAlbedo(GRID, hours_since_snowfall)

        # Update roughness length
        z0 = updateRoughness(GRID, hours_since_snowfall)

        # Merge grid layers, if necessary
        GRID.update_grid(merging_level)

        # Solve the heat equation
        solveHeatEquation(GRID, dt)

        # Find new surface temperature
        fun, surface_temperature, lw_radiation_in, lw_radiation_out, sensible_heat_flux, latent_heat_flux, \
            ground_heat_flux, sw_radiation_net, rho, Lv, Cs, q0, q2, qdiff, phi \
            = update_surface_temperature(GRID, alpha, z0, temperature_2m[t], relative_humidity[t], cloud_cover[t], air_pressure[t], solar_radiation[t], wind_speed[t])

        # Surface fluxes [m w.e.q.]
        if GRID.get_node_temperature(0) < zero_temperature:
            sublimation = max(latent_heat_flux / (1000.0 * lat_heat_sublimation), 0) * dt
            deposition = min(latent_heat_flux / (1000.0 * lat_heat_sublimation), 0) * dt
            evaporation = 0
            condensation = 0
        else:
            evaporation = max(latent_heat_flux / (1000.0 * lat_heat_vaporize), 0) * dt
            condensation = min(latent_heat_flux / (1000.0 * lat_heat_vaporize), 0) * dt
            sublimation = 0
            deposition = 0

        # Melt energy in [m w.e.q.]
        melt_energy = max(0, sw_radiation_net + lw_radiation_in + lw_radiation_out - ground_heat_flux -
                          sensible_heat_flux - latent_heat_flux)  # W m^-2 / J s^-1 ^m-2

        melt = melt_energy * dt / (1000 * lat_heat_melting)  # m w.e.q. (ice)

        # Remove melt height from surface and store as runoff (R)
        GRID.remove_melt_energy(melt + sublimation + deposition + evaporation + condensation)

        # Merge first layer, if too small (for model stability)
        GRID.merge_new_snow(merge_snow_threshold)

        # Account layer temperature due to penetrating SW radiation
        penetrating_radiation(GRID, sw_radiation_net, dt)

        # todo Percolation, fluid retention (liquid_water_content) & refreezing of melt water
        # and rain

        #print('size layer densities: ', GRID.layer_densities.shape," number nodess ",GRID.number_nodes)
        percolation(GRID, melt, dt)

        ### when freezing workt:
        # node_freezing, node_melting = percolation(GRID, melt, dt)

        # sum subsurface refreezing and melting
        # freezing = np.sum(node_freezing)
        # melting = np.sum(node_melting)

        # calculate mass balance [m w.e.]
        # mass_balance = (snowfall[t]*density_fresh_snow) - (melt + melting - freezing + sublimation + deposition + evaporation + condensation)

        albedo_all[t] = alpha
        condensation_all[t] = condensation
        depostion_all[t] = deposition
        evaporation_all[t] = evaporation
        ground_heat_flux_all[t] =ground_heat_flux
        longwave_in_all[t] = lw_radiation_in
        longwave_out_all[t] = lw_radiation_out
        latent_heat_flux_all[t] = latent_heat_flux
        # mass_balance_all[t] = mass_balance
        melt_heigt_all[t] = melt+sublimation+deposition+evaporation+condensation
        number_layers_all[t] = GRID.number_nodes
        # runoff_all[t] = runoff
        sensible_heat_flux_all[t] = sensible_heat_flux
        snowHeight_all[t] = np.sum((GRID.get_height()))
        shortwave_net_all[t] = sw_radiation_net
        sublimation_all[t] = sublimation
        surface_melt_all[t] = melt
        surface_temperature_all[t] = surface_temperature

        ###input variables save in same file for comparision; not needed in longtime???
        u2_all[t] = wind_speed[t]
        sw_in_all[t] = solar_radiation[t]
        T2_all[t] = temperature_2m[t]
        rH2_all[t] = relative_humidity[t]
        snowfall_all[t] = snowfall[t]
        pressure_all[t] = air_pressure[t]
        cloud_all[t] = cloud_cover[t]
        sh_all[t] = initial_snow_height[t]

        ###interim for investigations; not needed in longterm???
        rho_all[t] = rho
        Lv_all[t] = Lv
        Cs_all[t] = Cs
        q0_all[t] = q0
        q2_all[t] = q2
        qdiff_all[t] = qdiff
        phi_all[t] = phi

    del GRID, air_pressure, alpha, cloud_cover, condensation, deposition, evaporation, fun, gradient, \
        ground_heat_flux, hours_since_snowfall, i, initial_snow_height, latent_heat_flux, layer_heights, \
        liquid_water_content, lw_radiation_in, lw_radiation_out, melt, melt_energy, relative_humidity, rho, \
        sensible_heat_flux, snow_height, snowfall, solar_radiation, sublimation, surface_temperature, \
        sw_radiation_net, t, temperature_2m, temperature_surface, wind_speed, z0

    return albedo_all, condensation_all, depostion_all, evaporation_all, ground_heat_flux_all, \
        longwave_in_all, longwave_out_all, latent_heat_flux_all, melt_heigt_all, number_layers_all, \
        sensible_heat_flux_all, snowHeight_all, shortwave_net_all, sublimation_all, surface_melt_all, \
        surface_temperature_all, u2_all, sw_in_all, T2_all, rH2_all, snowfall_all, pressure_all, cloud_all, \
        sh_all, rho_all, Lv_all, Cs_all, q0_all, q2_all, qdiff_all, phi_all